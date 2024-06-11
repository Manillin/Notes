#include <iostream>
using namespace std;

int main()
{

    int a = 5;
    int b = 10;

    int *ptr = &a;
    int *ptr2 = ptr;

    // int *ptr3 = *ptr2; -> sbagliato !!!

    int **ptrDue = &ptr;

    int *ptr3 = *ptrDue;
}