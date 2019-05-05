#define OMPI_SKIP_MPICXX //I recommend not using C++ bindings
#include <mpi.h>
#include <omp.h>
#include <array>
#include <memory>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <cassert>

std::vector<std::size_t> g_node_counts;

std::uint32_t GetNodeId()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

std::uint32_t GetNumNodes()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

bool MasterNode()
{
    return GetNodeId() == 0u;
}

void Barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
}

std::uint32_t GetNodeIdFromElementId(std::size_t id)
{
    std::size_t count = 0;
    for (auto i = 0u; i < g_node_counts.size(); ++i)
    {
        count += g_node_counts[i];
        if (id < count)
        {
            return i;
        }
    }
    assert(0);
    return -1;
}

std::size_t GetLocalElementId(std::size_t global_id)
{
    std::size_t result = global_id;
    for (auto i = 1u; i <= GetNodeIdFromElementId(global_id); ++i)
    {
        result -= g_node_counts[i - 1];
    }

    return result;
}

void heapify(std::uint32_t * arr, int n)
{
    for (int i = 1; i < n; i++)
    {
        // if child is bigger than parent
        if (arr[i] > arr[(i - 1) / 2])
        { 
            int j = i;
      
            // swap child and parent until
            // parent is smaller 
            while (arr[j] > arr[(j - 1) / 2])
            { 
                std::swap(arr[j], arr[(j - 1) / 2]);
                j = (j - 1) / 2;
            }
        }
    }
}

void sort(
    std::uint32_t * data,
    int local_size,
    int global_size
)
{
    // Build heap (rearrange array)
    heapify(data, local_size);

    static const std::uint32_t sorted_marker = ~0u;

    std::size_t num_tree_elements = local_size;

    // Position where sorted part of the data starts
    for (std::size_t sorted_position = global_size - 1; sorted_position > 0; --sorted_position)
    {
        // Node id where sorted data starts
        std::uint32_t sorted_node_id = GetNodeIdFromElementId(sorted_position);

        // If number of tree elements is zero, the node is sorted
        bool is_node_sorted = (num_tree_elements == 0);

        // Gather roots
        std::vector<std::uint32_t> roots(GetNumNodes());

        // If a node is sorted, send a special marker that will be ignored
        std::uint32_t send_root = is_node_sorted ? sorted_marker : data[0];
        MPI_Allgather(&send_root, 1, MPI_UNSIGNED, roots.data(), 1, MPI_UNSIGNED, MPI_COMM_WORLD);

        // Find a node with max root
        std::uint32_t max_root_node_id = std::max_element(roots.begin(), roots.end(),
            [](std::uint32_t a, std::uint32_t b)
        {
            if (a == sorted_marker) return true;
            else if (b == sorted_marker) return false;
            else return a < b;
        }) - roots.begin();

        Barrier();

        // Check if we need to swap data between nodes
        if (max_root_node_id != sorted_node_id)
        {
            std::uint32_t recv_data;

            // Here we need to swap values:
            // data[0] from node with max_root_node_id
            // And data[GetLocalElementId(sorted_position)] from node with sorted_node_id

            // 1. Send data[0] from max_root_node_id to sorted_node_id
            if (GetNodeId() == max_root_node_id)
            {
                MPI_Send(&data[0], 1, MPI_UNSIGNED, sorted_node_id, 0, MPI_COMM_WORLD);
            }
            else if (GetNodeId() == sorted_node_id)
            {
                MPI_Status status;
                MPI_Recv(&recv_data, 1, MPI_UNSIGNED, max_root_node_id, 0, MPI_COMM_WORLD, &status);
            }

            // 2. Send data[GetLocalElementId(sorted_position)] form sorted_node_id to max_root_node_id
            if (GetNodeId() == sorted_node_id)
            {
                MPI_Send(&data[GetLocalElementId(sorted_position)], 1, MPI_UNSIGNED, max_root_node_id, 0, MPI_COMM_WORLD);
            }
            else if (GetNodeId() == max_root_node_id)
            {
                MPI_Status status;
                MPI_Recv(&recv_data, 1, MPI_UNSIGNED, sorted_node_id, 0, MPI_COMM_WORLD, &status);
            }

            // 3. Swap recv_data and data[GetLocalElementId(sorted_position)] on sorted_node_id
            if (GetNodeId() == sorted_node_id)
            {
                std::swap(recv_data, data[GetLocalElementId(sorted_position)]);
                // Decrease the number of tree elements
                --num_tree_elements;
                // And heapify remaining values inside the node
                heapify(data, num_tree_elements);
            }

            // 4. Swap recv_data and data[0] on max_root_node_id
            if (GetNodeId() == max_root_node_id)
            {
                std::swap(recv_data, data[0]);
                // And heapify
                heapify(data, num_tree_elements);
            }
        }
        else if (GetNodeId() == max_root_node_id)
        {
            // Swap the root with the last sorted position
            std::swap(data[0], data[GetLocalElementId(sorted_position)]);
            // Decrease the number of tree elements
            --num_tree_elements;
            // And heapify remaining values inside the node
            heapify(data, num_tree_elements);
        }

    }

}

auto GenerateData(std::size_t data_size)
{
    auto data = std::unique_ptr< std::uint32_t[] >(new std::uint32_t[data_size]);

    srand(GetNodeId());
    std::generate_n(data.get(), data_size, []()
    {
        return rand();
    });

    return std::move(data);
}

std::pair< bool, bool > is_sorted(const std::uint32_t * begin, const std::uint32_t * end, const MPI_Comm comm)
{
    auto mpi_type = MPI_INT;
    // Массив на узле отсортирован?
    auto inner_result = std::is_sorted(begin, end);
    auto num_nodes = GetNumNodes();

    auto outer_result = true;
    {
        // Собираем результаты о сортировках
        auto result = static_cast<std::uint32_t>(inner_result);
        auto others_results = std::unique_ptr< std::uint32_t[] >(new std::uint32_t[num_nodes]);
        MPI_Allgather(&result, 1, mpi_type, others_results.get(), 1, mpi_type, comm);
        auto others_total = std::all_of(others_results.get(), others_results.get() + num_nodes,
            [](const auto lhs)
        {
            return lhs != 0u;
        });
        outer_result &= others_total;
    }

    {
        constexpr auto mm_size = 2;
        // Минимум и максимум
        auto mm = std::array< std::uint32_t, mm_size >{*begin, *(begin + (std::distance(begin, end) - 1))};
        // Собираем данные о максимумах и минимумов со всех узлов
        auto others_mm = std::unique_ptr< std::uint32_t[] >(new std::uint32_t[num_nodes * mm_size]);
        MPI_Allgather(mm.data(), mm_size, mpi_type, others_mm.get(), mm_size, mpi_type, comm);
        // Если пары [min max] [min max] ... упорядочены,
        // то считаем данные отсортированными
        auto others_mm_total = std::is_sorted(others_mm.get(), others_mm.get() + num_nodes * mm_size);
        outer_result &= others_mm_total;
    }
    return { inner_result, outer_result };
}

int main(int argc, char ** argv)
{
    auto provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

    auto num_nodes = GetNumNodes();
    auto node_id = GetNodeId();

    g_node_counts.resize(num_nodes);

    if (argc != 2)
    {
        if (node_id == 0)
        {
            std::cerr << "Please, read README before using" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // Get total size from cmd line
    auto global_size = atoi(argv[1]);

    // Calculate data size per node
    const std::size_t local_size = (global_size / num_nodes) + ((node_id != (num_nodes - 1)) ? 0 : (global_size % num_nodes));
    Barrier();

    MPI_Allgather(&local_size, sizeof(std::size_t), MPI_CHAR, g_node_counts.data(), sizeof(std::size_t), MPI_CHAR, MPI_COMM_WORLD);

    // Generate data
    auto data = GenerateData(local_size);

    // Issue a barrier
    Barrier();

    // Sort
    sort(data.get(), local_size, global_size);

    // Issue a barrier
    Barrier();

    // Check for sorted
    auto result = is_sorted(data.get(), data.get() + local_size, MPI_COMM_WORLD);
    auto inner = result.first;
    auto outer = result.second;
    if (inner && outer)
    {
        if (node_id == 0)
        {
            std::clog << "Data is sorted" << std::endl;
        }
    }
    else
    {
        if (node_id == 0)
        {
            std::clog << "Data is unsorted" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
