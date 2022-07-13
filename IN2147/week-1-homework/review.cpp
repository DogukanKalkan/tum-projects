#include <cstring>
#include <iostream>
#include "vv-aes.h"
#include <map>
#include <algorithm>



int main(){
    std::map<int,int> myMap;

    for (int i = 0; i < UNIQUE_CHARACTERS; i++){
        myMap[originalCharacter[i]] = i;
    }
    /*
    for (std::map<int,int>::iterator it=myMap.begin(); it!=myMap.end(); ++it){
        std::cout << it->first << " => " << it->second << '\n';
    }
    std::cout << myMap.size() << std::endl;

    std::cout << myMap.find(100)->second << " " <<  myMap.find(100)->first << std::endl;
    */

    /*** STEP 1
    int map[UNIQUE_CHARACTERS];

    for(int i = 0; i < UNIQUE_CHARACTERS; i++){
        map[originalCharacter[i]] = i;
    }
    for(int i: map){
        std::cout << i << std::endl;
    }
    ***/

    std::cout << efficient_power(2, 4) << std::endl;
    std::cout << efficient_power(3, 5) << std::endl;
    std::cout << efficient_power(1, 5) << std::endl;
    std::cout << efficient_power(99, 5) << std::endl;
}


