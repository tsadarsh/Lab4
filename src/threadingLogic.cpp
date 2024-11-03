/*
Author: Adarsh Thirugnanasambandam Sriuma
Class: ECE6122 (A)
Last Date Modified: Oct 7, 2024

Description:

Process file for std::thread(s) Logic implementation.
*/

#include <thread>
#include <vector>

/**
 * populateNextGen - Calls OMP process on sequential logic
 *
 * @curr_gen: Pointer to current generation cells container
 * 
 * @next_gen: Pointer to next generation cells container
 * 
 * @job: Current generation cell indexes that is allocated to the thread invoking this function.
 */

void populateNextGen(std::vector<std::vector<int>>* curr_gen, std::vector<std::vector<int>>* next_gen, std::vector<std::vector<int>> job)
{
    for (int iJob = 0; iJob < job.size(); iJob++)
    {
        int i_row = job[iJob][0];
        int i_col = job[iJob][1];

        int nearby_score = (*curr_gen)[i_row - 1][i_col - 1] + (*curr_gen)[i_row - 1][i_col] + (*curr_gen)[i_row - 1][i_col + 1] + (*curr_gen)[i_row][i_col - 1] \
                    + (*curr_gen)[i_row][i_col + 1] + (*curr_gen)[i_row + 1][i_col - 1] + (*curr_gen)[i_row + 1][i_col + 1];

        // dead cell: check if birth possible
        if ((*curr_gen)[i_row][i_col] == 0)
        {
            if( nearby_score == 3)
                (*next_gen)[i_row][i_col] = 1;
        }

        // alive cell: check death by isolation, death by overcrowding, or survival
        else 
        {
            // death by isolation
            if (nearby_score <= 1)
                (*next_gen)[i_row][i_col] = 0;
            
            // death by overpopulation
            else if ( nearby_score >= 4)
                (*next_gen)[i_row][i_col] = 0;
            
            else{
                // cell survives, so do nothing
                (*next_gen)[i_row][i_col] = 1;
            }
        }
    }
}

/**
 * updateCurrGen - Updates the next generation cells container
 *
 * @curr_gen: Pointer to current generation cells container
 * 
 * @next_gen: Pointer to next generation cells container
 * 
 * @job: Current generation cell indexes that is allocated to the thread invoking this function.
 */

void updateCurrGen(std::vector<std::vector<int>>* curr_gen, std::vector<std::vector<int>>* next_gen, std::vector<std::vector<int>> job)
{
    for (int iJob = 0; iJob < job.size(); iJob++)
    {
        int i_row = job[iJob][0];
        int i_col = job[iJob][1];

        (*curr_gen)[i_row][i_col] = (*next_gen)[i_row][i_col];
    }
}

/**
 * THRD_Process - Starting point to THRD process. Creates requested number of threads and updates next generation.
 *
 * @curr_gen: Pointer to current generation cells container
 * 
 * @next_gen: Pointer to next generation cells container
 * 
 * @num_of_threads: Number of OMP threads to spawn
 */

void THRD_Process(std::vector<std::vector<int>>* curr_gen, std::vector<std::vector<int>>* next_gen, int num_of_threads, std::vector<std::vector<std::vector<int>>>* quotas)
{
    std::vector<std::thread> thread_pool_1;

    for (int iThread = 0; iThread < (*quotas).size(); iThread++)
    {
        thread_pool_1.push_back(std::thread(populateNextGen, std::ref(curr_gen), std::ref(next_gen), (*quotas)[iThread]));
    }
    
    for (int iThread = 0; iThread < thread_pool_1.size(); iThread++)
    {
        if (thread_pool_1[iThread].joinable())
            thread_pool_1[iThread].join();
    }

    std::vector<std::thread> thread_pool_2;
    for (int iThread = 0; iThread < (*quotas).size(); iThread++)
    {
        thread_pool_2.push_back(std::thread(updateCurrGen, std::ref(curr_gen), std::ref(next_gen), (*quotas)[iThread]));
    }

    for (int iThread = 0; iThread < thread_pool_2.size(); iThread++)
    {
        if (thread_pool_2[iThread].joinable())
            thread_pool_2[iThread].join();
    }
}