#include <iostream>

int main()
{
    int x = 10;
    auto lambda = [&x]()
    {
        x = 1;
    };

    auto lambda2 = [x]()
    {
        std::cout << "Valore di x: " << x << std::endl;
    };
    lambda();
    lambda2();
    std::cout << x << std::endl;
}
