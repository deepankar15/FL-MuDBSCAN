#include <mpi.h>
#include <vector>
#include <string>
#include <libxl.h>
#include <random>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include "Data.h"

using namespace libxl;

// DBSCAN function declaration (implemented elsewhere)
std::vector<std::vector<std::vector<double>>> micro_dbscan(
    const std::vector<std::vector<double>>& points, 
    double eps, 
    int min_pts
);

struct ClusterDescriptor {
    std::vector<double> centroid;
    int point_count;
    double max_radius;
};

// MPI Type creation for ClusterDescriptor
MPI_Datatype create_descriptor_type(int dimensions) {
    MPI_Datatype dtype;
    MPI_Aint displacements[3];
    int blocklengths[3] = {dimensions, 1, 1};
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_DOUBLE};
    
    displacements[0] = offsetof(ClusterDescriptor, centroid);
    displacements[1] = offsetof(ClusterDescriptor, point_count);
    displacements[2] = offsetof(ClusterDescriptor, max_radius);
    
    MPI_Type_create_struct(3, blocklengths, displacements, types, &dtype);
    MPI_Type_commit(&dtype);
    return dtype;
}

// Poisson Disk Sampling (Bridson's algorithm)
std::vector<std::vector<double>> generate_pds(double width, double height, 
                                            double min_dist, uint32_t seed) {
    std::vector<std::vector<double>> points;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    const double cell_size = min_dist / std::sqrt(2);
    const int grid_width = static_cast<int>(width / cell_size);
    const int grid_height = static_cast<int>(height / cell_size);

    std::vector<std::vector<int>> grid(grid_height, 
                                     std::vector<int>(grid_width, -1));
    std::vector<std::vector<double>> process_list;

    // Initial point
    std::vector<double> first = {dist(gen)*width, dist(gen)*height};
    points.push_back(first);
    process_list.push_back(first);

    while (!process_list.empty()) {
        std::uniform_int_distribution<size_t> idx_dist(0, process_list.size()-1);
        size_t idx = idx_dist(gen);
        auto point = process_list[idx];
        process_list.erase(process_list.begin() + idx);

        for (int i = 0; i < 30; ++i) {
            double angle = 2 * M_PI * dist(gen);
            double radius = min_dist * (1 + dist(gen));
            std::vector<double> new_point = {
                point[0] + radius * std::cos(angle),
                point[1] + radius * std::sin(angle)
            };

            if (new_point[0] < 0 || new_point[0] >= width ||
                new_point[1] < 0 || new_point[1] >= height) continue;

            int grid_x = static_cast<int>(new_point[0] / cell_size);
            int grid_y = static_cast<int>(new_point[1] / cell_size);

            bool valid = true;
            for (int x = std::max(0, grid_x-2); x < std::min(grid_width, grid_x+3); ++x) {
                for (int y = std::max(0, grid_y-2); y < std::min(grid_height, grid_y+3); ++y) {
                    if (grid[y][x] != -1) {
                        double dx = new_point[0] - points[grid[y][x]][0];
                        double dy = new_point[1] - points[grid[y][x]][1];
                        if (std::hypot(dx, dy) < min_dist) {
                            valid = false;
                            break;
                        }
                    }
                }
                if (!valid) break;
            }

            if (valid) {
                points.push_back(new_point);
                process_list.push_back(new_point);
                grid[grid_y][grid_x] = points.size() - 1;
            }
        }
    }
    return points;
}

// Excel data reader
std::vector<std::vector<double>> read_excel_data(const std::string& filename) {
    std::vector<std::vector<double>> data;
    Book* book = xlCreateBook();
    
    if(book->load(filename.c_str())) {
        Sheet* sheet = book->getSheet(0);
        if(sheet) {
            for(int row = sheet->firstRow(); row < sheet->lastRow(); ++row) {
                std::vector<double> point;
                for(int col = sheet->firstCol(); col < sheet->lastCol(); ++col) {
                    if(sheet->cellType(row, col) == CELLTYPE_NUMBER) {
                        point.push_back(sheet->readNum(row, col));
                    }
                }
                if(!point.empty()) data.push_back(point);
            }
        }
    }
    book->release();
    return data;
}

// Cluster descriptor generator
std::vector<ClusterDescriptor> create_descriptors(
    const std::vector<std::vector<std::vector<double>>>& clusters,
    const std::vector<std::vector<double>>& pds_points
) {
    std::vector<ClusterDescriptor> descriptors;
    
    for(const auto& cluster : clusters) {
        ClusterDescriptor desc;
        desc.point_count = cluster.size();
        
        // Calculate centroid
        std::vector<double> centroid(pds_points[0].size(), 0.0);
        for(const auto& point : cluster) {
            for(size_t i=0; i<point.size(); ++i) {
                centroid[i] += point[i];
            }
        }
        for(auto& val : centroid) val /= cluster.size();
        desc.centroid = centroid;
        
        // Calculate max radius
        double max_radius = 0.0;
        for(const auto& point : cluster) {
            double dist = 0.0;
            for(size_t i=0; i<point.size(); ++i) {
                dist += pow(point[i] - centroid[i], 2);
            }
            max_radius = std::max(max_radius, sqrt(dist));
        }
        desc.max_radius = max_radius;
        
        descriptors.push_back(desc);
    }
    return descriptors;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int DIMENSIONS = 3;  // Number of columns in Excel files
    const double EPS = 1.5;
    const int MIN_PTS = 5;
    const double AREA_SIZE = 100.0;
    const double PDS_MIN_DIST = 5.0;
    const int MAX_ITERATIONS = 10;
    
    MPI_Datatype descriptor_type = create_descriptor_type(DIMENSIONS);
    
    if(rank == 0) { // Server
        uint32_t current_seed = 12345;
        int previous_cluster_count = 0;
        
        for(int iter=0; iter<MAX_ITERATIONS; ++iter) {
            MPI_Bcast(&current_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            
            // Collect descriptors from clients
            std::vector<ClusterDescriptor> all_descriptors;
            for(int client=1; client<size; ++client) {
                int count;
                MPI_Recv(&count, 1, MPI_INT, client, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                std::vector<ClusterDescriptor> client_descs(count);
                MPI_Recv(client_descs.data(), count, descriptor_type, client, 
                       0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                all_descriptors.insert(all_descriptors.end(), 
                                     client_descs.begin(), client_descs.end());
            }
            
            // Convert to points for global clustering
            std::vector<std::vector<double>> global_points;
            for(const auto& desc : all_descriptors) {
                global_points.push_back(desc.centroid);
            }
            
            // Perform global clustering
            auto global_clusters = micro_dbscan(global_points, EPS*1.5, MIN_PTS);
            
            // Generate new seed
            std::random_device rd;
            current_seed = rd();
            
            // Check convergence
            if(iter > 0 && global_clusters.size() == previous_cluster_count) break;
            previous_cluster_count = global_clusters.size();
        }
    }
    else { // Client
        std::string filename = "client_" + std::to_string(rank) + ".xls";
        auto local_data = read_excel_data(filename);
        
        for(int iter=0; iter<MAX_ITERATIONS; ++iter) {
            uint32_t seed;
            MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            
            // Generate PDS points
            auto pds_points = generate_pds(AREA_SIZE, AREA_SIZE, PDS_MIN_DIST, seed);
            
            // Perform local clustering
            auto clusters = micro_dbscan(local_data, EPS, MIN_PTS);
            
            // Create descriptors
            auto descriptors = create_descriptors(clusters, pds_points);
            
            // Send to server
            int count = descriptors.size();
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(descriptors.data(), count, descriptor_type, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Type_free(&descriptor_type);
    MPI_Finalize();
    return 0;
}
