#define OMPI_SKIP_MPICXX //I recommend not using C++ bindings
#include <mpi.h>
#include <omp.h>
#include <array>
#include <limits>
#include <memory>
#include <chrono>
#include <random>
#include <ostream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <functional>

// Тип сортируемых значений (using <псевдоним> = <тип>)
// Не советую менять, так как типу unsigned int
// соответствует MPI_INT. Если поменяете, то имейте ввиду,
// что необходимо передать соответствующий тип в
// функции MPI при пересылке данных
using int_t = unsigned int;
namespace utility
{
    // Процесс и его описание: количество потоков, ранк
    // и размер коммуникатора (по-умолчанию коммуникатор MPI_COMM_WORLD)
    struct Process
    {
        int threads, rank, size;
        MPI_Comm comm;
        explicit Process(MPI_Comm c);
    };

    std::ostream & operator<<(std::ostream & out, const Process & rhs);

    std::pair< bool, bool > is_sorted(
        const int_t * begin,
        const int_t * end,
        const MPI_Comm comm
    );

    int GetRank()
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }

    void heapify(int_t arr[], int n, int i)
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
        int_t * data,
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
            std::cout << "Rank: " << GetRank() << ": ";
            for (auto it = data; it != data + size; ++it)
            {
                std::cout << *it << " ";
            }
            std::cout << std::endl;
        }
    }
}

int main(int argc, char ** argv)
{
    auto provided = 0;
    //Инициализация MPI. std::addressof эквивалентно оператору '&'
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    // Описатель процесса
    const auto pr = utility::Process(MPI_COMM_WORLD);
    // Два аргумента ./task   <amount of numbers>
    //   первый(имя) ^ второй ^
    if (argc != 2)
    {
        if (pr.rank == 0)
        {
            std::cerr << "Please, read README before using" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
    // Можно вывести в поток
    //if (pr.rank == 0)
    //{
    //    std::cout << pr << std::endl;
    //}
    // Для второго аргумента: строка -> число
    auto amount = atoi(argv[1]);
    // Одинаковое количество элементов на каждом узле с точности до количества процессов/узлов
    const auto size_per_rank = (amount / pr.size) + ((pr.rank != (pr.size - 1)) ? 0 : (amount % pr.size));
    // Выделение памяти под данные
    auto data = std::unique_ptr< int_t[] >(new int_t[size_per_rank]);
    {
        // Генерация случайных чисел: левая и правая границы,
        // они же минимально и максимально возможные значения для типа int_t
        constexpr auto left_border = std::numeric_limits< int_t >::min();
        constexpr auto right_border = std::numeric_limits< int_t >::max();
        // Генератор. Вызываем как функцию
        namespace ch = std::chrono;
        auto dice = std::bind(
            std::uniform_int_distribution< int_t >{left_border, right_border},
            std::default_random_engine
            {
              ch::duration_cast<ch::duration< int_t >>(ch::system_clock::now().time_since_epoch()).count()
            }
        );
        srand(utility::GetRank());
        std::generate_n(data.get(), size_per_rank,
            [&](void)
        {
            // dice() вернёт "случайное" число
            return rand() % 100;//dice();
        });
    }
    // Вызов сортировки
    utility::sort(data.get(), size_per_rank, MPI_COMM_WORLD);
    // Проверка отсортированности: в пределах узла (inner) и по отношению к другим узлам (outer)
    auto result = utility::is_sorted(data.get(), data.get() + size_per_rank, MPI_COMM_WORLD);
    auto inner = result.first;
    auto outer = result.second;
    if (inner && outer)
    {
        if (pr.rank == 0)
        {
            std::clog << "Data is sorted" << std::endl;
        }
    }
    else
    {
        if (pr.rank == 0)
        {
            std::clog << "Data is unsorted" << std::endl;
        }
    }
    // Не забываем вызвать
    MPI_Finalize();
    return 0;
}

namespace utility
{
    Process::Process(MPI_Comm c) :
        threads(omp_get_max_threads()),
        rank(0),
        size(0),
        comm(c)
    {
        auto initialized = 0;
        // Проверяем, что инициализирован MPI
        MPI_Initialized(&initialized);
        if (initialized)
        {
            // Получаем ранк
            MPI_Comm_rank(comm, &rank);
            // Получаем размер коммуникатора
            MPI_Comm_size(comm, &size);
        }
    }

    std::ostream & operator<<(std::ostream & out, const Process & rhs)
    {
        // Выводим описатель процесса в консоль
        out << "(:threads " << rhs.threads;
        out << ":rank " << rhs.rank;
        out << ":size " << rhs.size;
        out << ":)";
        return out;
    }

    std::pair< bool, bool > is_sorted(const int_t * begin, const int_t * end, const MPI_Comm comm)
    {
        // Компаратор по-умолчанию
        auto comp = std::less< int_t >{};
        auto mpi_type = MPI_INT;
        // Массив на узле отсортирован?
        auto inner_result = std::is_sorted(begin, end, comp);
        auto pr = Process{ comm };

        auto outer_result = true;
        {
            // Собираем результаты о сортировках
            auto result = static_cast<int_t>(inner_result);
            auto others_results = std::unique_ptr< int_t[] >(new int_t[pr.size]);
            MPI_Allgather(&result, 1, mpi_type, others_results.get(), 1, mpi_type, comm);
            auto others_total = std::all_of(others_results.get(), others_results.get() + pr.size,
                [](const auto lhs)
            {
                return lhs != 0u;
            });
            outer_result &= others_total;
        }

        {
            constexpr auto mm_size = 2;
            // Минимум и максимум
            auto mm = std::array< int_t, mm_size >{*begin, *(begin + (std::distance(begin, end) - 1))};
            // Собираем данные о максимумах и минимумов со всех узлов
            auto others_mm = std::unique_ptr< int_t[] >(new int_t[pr.size * mm_size]);
            MPI_Allgather(mm.data(), mm_size, mpi_type, others_mm.get(), mm_size, mpi_type, comm);
            // Если пары [min max] [min max] ... упорядочены,
            // то считаем данные отсортированными
            auto others_mm_total = std::is_sorted(others_mm.get(), others_mm.get() + pr.size * mm_size, comp);
            outer_result &= others_mm_total;
        }
        return { inner_result, outer_result };
    }
}
