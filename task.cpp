#define OMPI_SKIP_MPICXX //I recommend not using C++ bindings
#include <mpi.h>
#include <omp.h>
#include <array>
#include <memory>
#include <iostream>
#include <iterator>
#include <algorithm>

int GetNodeId()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int GetNumNodes()
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

void heapify(std::uint32_t arr[], int n, int i)
{
    int largest = i; // Initialize largest as root 
    int l = 2 * i + 1; // left = 2*i + 1 
    int r = 2 * i + 2; // right = 2*i + 2 

    // If left child is larger than root 
    if (l < n && arr[l] > arr[largest])
        largest = l;

    // If right child is larger than largest so far 
    if (r < n && arr[r] > arr[largest])
        largest = r;

    // If largest is not root 
    if (largest != i)
    {
        std::swap(arr[i], arr[largest]);

        // Recursively heapify the affected sub-tree 
        heapify(arr, n, largest);
    }
}

void sort(
    std::uint32_t * data,
    int size,
    const MPI_Comm comm
)
{
    // Build heap (rearrange array) 
    for (int i = size / 2 - 1; i >= 0; i--)
        heapify(data, size, i);

    // One by one extract an element from heap 
    for (int i = size - 1; i >= 0; i--) {
        // Move current root to end 
        std::swap(data[0], data[i]);

        // call max heapify on the reduced heap 
        heapify(data, i, 0);
    }

    //if (GetRank() == 0)
    {
        std::cout << "Node " << GetNodeId() << ": ";
        for (auto it = data; it != data + size; ++it)
        {
            std::cout << *it << " ";
        }
        std::cout << std::endl;
    }
}

auto GenerateData(std::size_t data_size)
{
    auto data = std::unique_ptr< std::uint32_t[] >(new std::uint32_t[data_size]);

    srand(GetNodeId());
    std::generate_n(data.get(), data_size, []()
    {
        return rand() % 100;
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
    auto total_size = atoi(argv[1]);

    // Calculate data size per node
    const auto size_per_node = (total_size / num_nodes) + ((node_id != (num_nodes - 1)) ? 0 : (total_size % num_nodes));
    // Generate data
    auto data = GenerateData(size_per_node);

    // Sort
    sort(data.get(), size_per_node, MPI_COMM_WORLD);

    // Check for sorted
    auto result = is_sorted(data.get(), data.get() + size_per_node, MPI_COMM_WORLD);
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
