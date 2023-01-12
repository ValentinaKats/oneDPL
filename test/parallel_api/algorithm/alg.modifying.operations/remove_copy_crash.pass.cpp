// ====------ remove_copy.cu------------- *- CUDA -* -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// ===-------------------------------------------------------------------===//

#include "support/test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

//#include <oneapi/dpl/execution>
//#include <oneapi/dpl/algorithm>
#define DPCT_USM_LEVEL_NONE
//#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <dpct/dpl_utils.hpp>

void
test_1()
{ // host iterator
    const int N = 6;
    int A[N] = {-2, 0, -1, 0, 1, 2};
    int B[N - 2];
    int ans[N - 2] = {-2, -1, 1, 2};
    std::vector<int> V(A, A + N);
    std::vector<int> result(B, B + N - 2);

    oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V.begin(), V.end(), result.begin(), 0);
    for (int i = 0; i < N - 2; i++)
    {
        if (result[i] != ans[i])
        {
            printf("test_1 run failed\n");
            exit(-1);
        }
    }

    printf("test_1 run passed!\n");
}

void
test_2()
{ // host iterator

    const int N = 6;
    int A[N] = {-2, 0, -1, 0, 1, 2};
    int B[N - 2];
    int ans[N - 2] = {-2, -1, 1, 2};
    std::vector<int> V(A, A + N);
    std::vector<int> result(B, B + N - 2);

    oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V.begin(), V.end(), result.begin(), 0);
    for (int i = 0; i < N - 2; i++)
    {
        if (result[i] != ans[i])
        {
            printf("test_2 run failed\n");
            exit(-1);
        }
    }

    printf("test_2 run passed!\n");
}

void
test_3()
{
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue(); // device iterator
    const int N = 6;
    int A[N] = {-2, 0, -1, 0, 1, 2};
    int B[N - 2];
    int ans[N - 2] = {-2, -1, 1, 2};
    dpct::device_vector<int> V(A, A + N);
    dpct::device_vector<int> result(B, B + N - 2);

    oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(q_ct1), V.begin(), V.end(), result.begin(), 0);
    for (int i = 0; i < N - 2; i++)
    {
        if (result[i] != ans[i])
        {
            printf("test_3 run failed\n");
            exit(-1);
        }
    }

    printf("test_3 run passed!\n");
}

void
test_4()
{ // host iterator

    const int N = 6;
    int A[N] = {-2, 0, -1, 0, 1, 2};
    int B[N - 2];
    int ans[N - 2] = {-2, -1, 1, 2};
    dpct::device_vector<int> V(A, A + N);
    dpct::device_vector<int> result(B, B + N - 2);

    oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()), V.begin(), V.end(),
                             result.begin(), 0);
    for (int i = 0; i < N - 2; i++)
    {
        if (result[i] != ans[i])
        {
            printf("test_4 run failed\n");
            exit(-1);
        }
    }

    printf("test_4 run passed!\n");
}

void
test_5()
{ // host iterator
    const int N = 6;
    int V[N] = {-2, 0, -1, 0, 1, 2};
    int result[N - 2];
    int ans[N - 2] = {-2, -1, 1, 2};

    if (dpct::is_device_ptr(V))
    {
        oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                                 dpct::device_pointer<int>(V), dpct::device_pointer<int>(V + N),
                                 dpct::device_pointer<int>(result), 0);
    }
    else
    {
        oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);
    };
    for (int i = 0; i < N - 2; i++)
    {
        if (result[i] != ans[i])
        {
            printf("test_5 run failed\n");
            exit(-1);
        }
    }

    printf("test_5 run passed!\n");
}

void
test_6()
{

    const int N = 6;
    int V[N] = {-2, 0, -1, 0, 1, 2};
    int result[N - 2];
    int ans[N - 2] = {-2, -1, 1, 2};

    if (dpct::is_device_ptr(V + N))
    {
        oneapi::dpl::remove_copy(oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
                                 dpct::device_pointer<int>(V), dpct::device_pointer<int>(V + N),
                                 dpct::device_pointer<int>(result), 0);
    }
    else
    {
        oneapi::dpl::remove_copy(oneapi::dpl::execution::seq, V, V + N, result, 0);
    };
    for (int i = 0; i < N - 2; i++)
    {
        if (result[i] != ans[i])
        {
            printf("test_6 run failed\n");
            exit(-1);
        }
    }

    printf("test_6 run passed!\n");
}

int
main()
{
    test_1();
    test_2();
    test_3(); // test_3 run failed when migrated with none-USM mode
    test_4();
    test_5();
    test_6();

    return 0;
}