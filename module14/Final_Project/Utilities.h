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
	return ((W*k)+H);
}

void PrintHuffmanCode(pixfreq<25> *pix_freq, int nodes)
{
   // Printing Huffman Codes
   printf("Huffmann Codes::\n\n");
   printf("pixel values -> Code\n\n");
   for (i = 0; i < nodes; i++) 
   {
      if (snprintf(NULL, 0, "%d", pix_freq[i].intensity) == 2)
         printf("  %d    -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
      else
         printf(" %d  -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
   }
}

#endif /* UTILITIES_H_ */
