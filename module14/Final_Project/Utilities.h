/*
 * Utilities.h
 *
 *  Created on: Apr 10, 2022
 *      Author: Wilson
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

// Defining Structures pixfreq
template<unsigned int N>
struct pixfreq
{
   int intensity, larrloc, rarrloc;
   float Freq;
   struct pixfreq<N> *left, *right;
   char code[N];
};


// Defining Structures
// huffcode
struct huffcode
{
   int intensity, arrloc;
   float Freq;
};

// function to find fibonacci number
int fib(int n)
{
   if (n <= 1)
      return n;
   return fib(n - 1) + fib(n - 2);
}

int index(int H, int W, int k) 
{
	return ((W*k)+H));
}

#endif /* UTILITIES_H_ */
