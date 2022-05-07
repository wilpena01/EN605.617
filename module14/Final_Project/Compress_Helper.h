#ifndef COMPRESS_HELPER_H_
#define COMPRESS_HELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Utilities.h"
#include <string>
#include <fstream>

using namespace std;

void readBMPFILE(int &width, int &height, int** &image)
{
    // load bmp image
    int i, j, temp = 0, offset = 2, bpp = 0;
    char file[] = "Lena.bmp";
    long bmpS = 0, bmpoff = 0;

    // Reading the BMP File
    FILE* inputImage;

    inputImage = fopen(file, "rb");
    if (inputImage == NULL)
    {
        cout<<"Error input File!!"<<endl;
        exit(1);
    }
    else
    {

        fseek(inputImage, offset, SEEK_SET);
        fread(&bmpS, 4, 1, inputImage); offset = 10;
        fseek(inputImage, offset, SEEK_SET);
        fread(&bmpoff, 4, 1, inputImage);
        fseek(inputImage, 18, SEEK_SET);
        fread(&width, 4, 1, inputImage);
        fread(&height, 4, 1, inputImage);
        fseek(inputImage, 2, SEEK_CUR);
        fread(&bpp, 2, 1, inputImage);
        fseek(inputImage, bmpoff, SEEK_SET);

        // Creating Image array
        image = (int**)malloc(height * sizeof(int*));

        for (i = 0; i < height; i++)
        {
            image[i] = (int*)malloc(width * sizeof(int*));
        }
        // Reading the inputImage
        // into the Image Array
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                fread(&temp, 3, 1, inputImage);

                // the Image is a
                // 24-bit BMP Image
                temp = temp & 0x0000FF;
                image[i][j] = static_cast<int>(temp);
            }
        }
    }
    fclose(inputImage);
}

void ocurrence(int* hist, int** image, int width, int height)
{
    // Finding the probability
    // of occurrence
    int n,k;
   
    for (n = 0; n < 256; n++)
        hist[n] = 0;

    for (n = 0; n < height; n++)
    {
        for (k = 0; k < width; k++)
        {
            hist[image[n][k]] += 1;
        }
    }
}

void nonZero_ocurrence(int* hist, int &node)
{
    // Finding number of
    // non-zero occurrences
    node=0;
    int n;
    for (n = 0; n < 256; n++)
      if (hist[n] != 0)
         node = node + 1;
}

void minProp(float &prob, int* hist, int width, int height)
{
    // Calculating minimum probability
    float currProb;
    int n;
    prob = 1.0;
    for (n = 0; n < 256; n++)
    {
        currProb = (hist[n] / (float)(height * width));
        if (currProb > 0 && currProb <= prob)
            prob = currProb;
    }
}

int MaxLength(float prob)
{
    // Calculating max length
    // of code word
    int n = 0;
    while ((1 / prob) > fib(n))
    {
        n++;
    }

    return n;
}

void InitStruct(pixfreq<25> *&pix_freq, huffcode* &huffcodes, 
                int* hist, int height, int width)
{
     // Initializing
   int n; int k=0;
   int MaxSize = height * width;
   float currProb;
   for (n = 0; n < 256; n++)
   {
      if (hist[n] != 0)
      {

         huffcodes[k].intensity = n;
         pix_freq[k].intensity  = n;
         huffcodes[k].arrloc    = k;
         currProb = (float)hist[n] / (float)MaxSize;
         pix_freq[k].Freq       = currProb;
         huffcodes[k].Freq      = currProb;
         pix_freq[k].left       = NULL;
         pix_freq[k].right      = NULL;
         pix_freq[k].code[0]    = '\0';
         k++;
      }
   }

}

void sortHist(huffcode* &huffcodes, int nodes)
{
    // Sorting the histogram
    int n, k;
    huffcode huff;

    // Sorting probability
    for (n = 0; n < nodes; n++)
    {
        for (k = n + 1; k < nodes; k++)
        {
            if (huffcodes[n].Freq < huffcodes[k].Freq)
            {
                huff = huffcodes[n];
                huffcodes[n] = huffcodes[k];
                huffcodes[k] = huff;
            }
        }
    }
}

void BuildTree(pixfreq<25> *&pix_freq, huffcode* &huffcodes, int nodes)
{
    // Building Huffman Tree
    float totalprob;
    int totalpix, z;
    int i = 0, j = 0;
    int n_node = nodes;

    while (i < nodes - 1)
    {

        // Adding the lowest two probabilities
        totalprob = huffcodes[nodes - i - 1].Freq + huffcodes[nodes - i - 2].Freq;
        totalpix  = huffcodes[nodes - i - 1].intensity + huffcodes[nodes - i - 2].intensity;

        // Appending to the pix_freq Array
        pix_freq[n_node].intensity = totalpix;
        pix_freq[n_node].Freq      = totalprob;
        pix_freq[n_node].left      = &pix_freq[huffcodes[nodes - i - 2].arrloc];
        pix_freq[n_node].right     = &pix_freq[huffcodes[nodes - i - 1].arrloc];
        pix_freq[n_node].code[0]   = '\0';
        z = 0;

        // Sorting and Updating the
        // huffcodes array simultaneously
        // New position of the combined node
        while (totalprob <= huffcodes[z].Freq)
            z++;

        // Inserting the new node
        // in the huffcodes array
        for (j = nodes; j >= 0; j--)
        {
            if (j == z)
            {
                huffcodes[j].intensity = totalpix;
                huffcodes[j].Freq = totalprob;
                huffcodes[j].arrloc = n_node;
            }
            else if (j > z)

            // Shifting the nodes below
            // the new node by 1
            // For inserting the new node
            // at the updated position k
            huffcodes[j] = huffcodes[j - 1];

        }
        i = i + 1;
        n_node = n_node + 1;
    }





}

void stradd(char* strptr, char* pcode, char add)
{
    // function to concatenate the words
   int i = 0;
   while (*(pcode + i) != '\0')
   {
      *(strptr + i) = *(pcode + i);
      i++;
   }
   if (add != '2')
   {
      strptr[i] = add;
      strptr[i + 1] = '\0';
   }
   else
      strptr[i] = '\0';
}

void AssignCode(pixfreq<25> *&pix_freq, int nodes, int totalnodes)
{
    // Assigning Code 
    int n;
    char left = '0';
    char right = '1';
    for (n = totalnodes - 1; n >= nodes; n--)
    {
        if (pix_freq[n].left != NULL)
        {
            stradd(pix_freq[n].left->code, pix_freq[n].code, left);
        }
        if (pix_freq[n].right != NULL)
        {
            stradd(pix_freq[n].right->code, pix_freq[n].code, right);
        }
    }
}

#endif /* COMPRESS_HELPER_H_ */