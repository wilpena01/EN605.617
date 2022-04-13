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


int codelen(char* code)
{
    // function to calculate word length
   int l = 0;
   while (*(code + l) != '\0')
      l++;
   return l;
}

void PrintHuffmanCode(pixfreq<25> *pix_freq, int nodes)
{
   // Printing Huffman Codes
   printf("Huffmann Codes::\n\n");
   printf("pixel values -> Code\n\n");
   for (int i = 0; i < nodes; i++) 
   {
      if (snprintf(NULL, 0, "%d", pix_freq[i].intensity) == 2)
         printf("  %d    -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
      else
         printf(" %d  -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
   }
}

void calBitLength(pixfreq<25> *pix_freq, int nodes)
{
   // Calculating Average Bit Length
   float avgbitnum = 0;
   for (int i = 0; i < nodes; i++)
      avgbitnum += pix_freq[i].Freq * codelen(pix_freq[i].code);
   printf("Average number of bits:: %f", avgbitnum);

}
#endif /* UTILITIES_H_ */
