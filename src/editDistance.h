#pragma once

#include <string>
#include <algorithm>

int EditDistance(std::string s1, std::string s2)
{
    // create two work vectors of integer distances
    int m = s1.size();
    int n = s2.size();
    int* v0 = new int[n + 1];
    int* v1 = new int[n + 1];

    // initialize v0 (the previous row of distances)
    // this row is A[0][i]: edit distance for an empty s
    // the distance is just the number of characters to delete from t
    for (int i = 0; i <= n; i++)
        v0[i] = i;

    for (int i = 0; i < m; i++)
    {

        // calculate v1 (current row distances) from the previous row v0

        // first element of v1 is A[i+1][0]
        //   edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i + 1;

        // use formula to fill in the rest of the row
        for (int j = 0; j < n; j++)
        {
            // calculating costs for A[i+1][j+1]
            int deletionCost = v0[j + 1] + 1;
            int insertionCost = v1[j] + 1;
            int substitutionCost;
            if (s1[i] == s2[j])
                substitutionCost = v0[j];
            else
                substitutionCost = v0[j] + 1;

            v1[j + 1] = std::min(std::min(deletionCost, insertionCost), substitutionCost);
        }
        // copy v1 (current row) to v0 (previous row) for next iteration
        // since data in v1 is always invalidated, a swap without copy could be more efficient
        for (int j = 0; j <= n; j++)
            v0[j] = v1[j];
    }
    // after the last swap, the results of v1 are now in v0

    delete[] v0;
    delete[] v1;

    return v0[n];
}

int EditDistanceCheckPrefix(std::string s1, std::string s2, int max_dst)
{
    // create two work vectors of integer distances
    int m = s1.size();
    int n = s2.size();
    int* v0 = new int[n + 1];
    int* v1 = new int[n + 1];

    // initialize v0 (the previous row of distances)
    // this row is A[0][i]: edit distance for an empty s
    // the distance is just the number of characters to delete from t
    for (int i = 0; i <= n; i++)
        v0[i] = i;

    for (int i = 0; i < m; i++)
    {

        // calculate v1 (current row distances) from the previous row v0

        // first element of v1 is A[i+1][0]
        //   edit distance is delete (i+1) chars from s to match empty t
        v1[0] = i + 1;

        // use formula to fill in the rest of the row
        for (int j = 0; j < n; j++)
        {
            // calculating costs for A[i+1][j+1]
            int deletionCost = v0[j + 1] + 1;
            int insertionCost = v1[j] + 1;
            int substitutionCost;
            if (s1[i] == s2[j])
                substitutionCost = v0[j];
            else
                substitutionCost = v0[j] + 1;

            v1[j + 1] = std::min(std::min(deletionCost, insertionCost), substitutionCost);
        }
        // copy v1 (current row) to v0 (previous row) for next iteration
        // since data in v1 is always invalidated, a swap without copy could be more efficient
        for (int j = 0; j <= n; j++)
            v0[j] = v1[j];

        // if prefix in max_dst
        if (v0[n] <= max_dst)
        {
            delete[] v0;
            delete[] v1;
            return true;
        }
    }

    delete[] v0;
    delete[] v1;

    return false;
}