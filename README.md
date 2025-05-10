# Задание по параллельному умножению матриц (расширенная версия)

**Курс:** Параллельное программирование и суперкомпьютерный кодизайн  
**Студент:** Мартиросян Арсен  

---

## Описание решения

В данной работе реализовано параллельное умножение матриц с использованием следующих подходов:

- **std::thread с std::atomic** — для распараллеливания по строкам.
- **OpenMP с reduction** — для распараллеливания вложенных циклов.
- **CUDA** — для реализации перемножения на видеокарте.
- **FlatMatrix** — оптимизированное представление матрицы в виде одномерного массива.
- **Транспонирование B** — для повышения производительности (особенно на CUDA и FlatMatrix).

---

## Организация проекта

Структура проекта включает следующие файлы:

```
.
├── main.cpp
├── Matrix.h
├── FlatMatrix.h
├── thread_atomic.h
├── openmp_reduction.h
├── cuda_mul.cu
├── cuda_mul.h
├── Makefile
```

---

## Matrix.h

Реализует класс `Matrix`, обеспечивающий:

- Стандартное хранение матрицы в виде `vector<vector<double>>`.
- Конструкторы с нулевой инициализацией и заполнением заданными значениями.
- Загрузка и сохранение матриц в файл.
- Последовательное умножение матриц (`Matrix::multiply(...)`).
- Сравнение матриц с заданной точностью.

---

## FlatMatrix.h

Реализует класс `FlatMatrix`, который хранит матрицу в виде одномерного массива `vector<double>` для повышения эффективности доступа к памяти, включая:

- Методы доступа по индексам: `operator()(i, j)`
- Метод `transpose()` — транспонирует матрицу на месте.
- Конструкторы и методы генерации случайной матрицы.

---

## thread_atomic.h

Функции:

```cpp
Matrix multiply_thread_atomic(const Matrix& A, const Matrix& B);
FlatMatrix multiply_flat_thread_atomic(FlatMatrix& A, FlatMatrix& B);
FlatMatrix multiply_flat_transposed_thread_atomic(FlatMatrix& A, FlatMatrix& B);
```

Описание:

- Используется `std::atomic<int>` как счётчик текущей строки.
- Каждый поток захватывает строку, умножает её на все столбцы матрицы B.
- В FlatMatrix-версиях используются прямой доступ к массиву и транспонирование B.

---

## openmp_reduction.h

Функции:

```cpp
Matrix multiply_openmp_reduction(const Matrix& A, const Matrix& B);
FlatMatrix multiply_flat_openmp_reduction(FlatMatrix& A, FlatMatrix& B);
FlatMatrix multiply_flat_transposed_openmp_reduction(FlatMatrix& A, FlatMatrix& B);
```

Описание:

- Внешний цикл размечается `#pragma omp parallel for`.
- Внутренний цикл — `#pragma omp parallel for reduction(+:sum)`.
- В транспонированных версиях перед умножением вызывается `B.transpose()`.

---

## cuda_mul.{h, cu}

Функции:

```cpp
FlatMatrix multiply_flat_cuda(FlatMatrix& A, FlatMatrix& B);
FlatMatrix multiply_flat_transposed_cuda(FlatMatrix& A, FlatMatrix& B);
```

Описание:

- CUDA-ядро запускается с сеткой `dim3(grid, block)` с ограничением на максимальные размеры.
- Используется транспонирование B для повышения локальности доступа к памяти.
- Вся память копируется на GPU и обратно.
- Результат сохраняется в FlatMatrix.

---

## main.cpp

Режимы работы:

1. **Тестовый режим:**
   - Загружаются матрицы `A` и эталон `C` из файлов.
   - Умножение `A*A` каждым методом.
   - Сравнение с эталоном: вывод OK/FAILED.

2. **Пользовательский режим:**
   - Генерация случайных матриц заданного размера.
   - Умножение выбранным методом.
   - Замер времени выполнения.

---

## Сборка проекта

Сборка выполняется командой:

```bash
make
```

Очистка:

```bash
make clean
```

---

## Запуск

### Тестовый режим

```bash
./matrix_mul
```

Пример:

```
Test mode results:
  atomic: OK
  openmp: OK
  flat_atomic: OK
  flat_openmp: OK
  cuda: OK
```

### Пользовательский режим

```bash
./matrix_mul <method> <size>
```

Где `<method>` — один из:
- `atomic`
- `openmp`
- `flat_atomic`
- `flat_openmp`
- `flat_transposed_atomic`
- `flat_transposed_openmp`
- `cuda`
- `cuda_transposed`

Пример:

```bash
./matrix_mul cuda_transposed 512
Execution time (cuda_transposed): 0.002431 seconds
```

---

## Результат работы программы

```bash
└─$ ./matrix_mul 
Test mode results:
  atomic: OK
  openmp: OK
  cuda:   OK
FlatMatrix test mode results:
  atomic_flat: OK
  openmp_flat: OK
  cuda_flat:   OK
FlatMatrix with transposed B test mode results:
  atomic_flat_transposed: OK
  openmp_flat_transposed: OK
  cuda_flat_transposed:   OK

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul atomic 2000
Execution time (atomic): 4.75548 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul atomic_flat 2000
Execution time (atomic_flat): 3.90875 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul atomic_flat_transposed 2000
Execution time (atomic_flat_transposed): 2.08463 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul openmp 2000
Execution time (openmp): 5.15875 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul openmp_flat 2000
Execution time (openmp_flat): 4.0734 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul openmp_flat_transposed 2000
Execution time (openmp_flat_transposed): 1.11311 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul cuda 5000
Execution time (cuda): 3.86408 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul cuda_flat 5000
Execution time (cuda_flat): 3.41432 seconds

┌──(dh㉿hostbomb-new)-[/mnt/c/Users/dh/Downloads/Martirosyan_matrix_task_with_cuda]
└─$ ./matrix_mul cuda_flat_transposed 5000
Execution time (cuda_flat_transposed): 1.47655 seconds
```

## Заключение

В рамках работы реализованы и протестированы 7 различных методов умножения матриц с использованием технологий параллельного программирования: std::thread, OpenMP и CUDA. Также применено представление FlatMatrix и оптимизация через транспонирование правой матрицы. Все методы протестированы, результаты корректны. Реализация модульна, структурирована и готова к масштабированию. Также, исходя из полученных результатов сделан вывод об эффективности применения различных методов параллелизации вычислений, согласно которому эффективность распределяется так:
**Векторы в векторе < одномерный массив < одномерный массив со второй транспонированной матрицей**