/*
Author: Adarsh Thirugnanasambandam Sriuma
Class: ECE6122 (A)
Last Date Modified: Oct 7, 2024

Description:

Process file for Sequential Logic implementation.
*/

#include <vector>

/**
 * SEQ_Process - Processes next generation cell values from current genration status
 *
 * @curr_gen: Pointer to current generation cells container
 * 
 * @next_gen: Pointer to next generation cells container
 */

void SEQ_Process(std::vector<std::vector<int>>* curr_gen, std::vector<std::vector<int>>* next_gen)
{
    for (int i_row=1; i_row < (*curr_gen).size()-1; i_row++)
    {
        for (int i_col=1; i_col < (*curr_gen)[i_row].size()-1; i_col++)
        {
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
    // copy next gen to curr generation
    for(int i_row = 1; i_row < (*next_gen).size()-1; i_row++)
    {
        for(int i_col = 1; i_col < (*next_gen)[i_row].size()-1; i_col++)
        {
            (*curr_gen)[i_row][i_col] = (*next_gen)[i_row][i_col];
        }
    }
}