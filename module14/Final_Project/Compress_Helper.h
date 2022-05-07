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
   int totpix = height * width;
   float tempprob;
   for (n = 0; n < 256; n++)
   {
      if (hist[n] != 0)
      {

         // pixel intensity value
         huffcodes[k].intensity = n;
         pix_freq[k].intensity = n;

         // location of the node
         // in the pix_freq array
         huffcodes[k].arrloc = k;

         // probability of occurrence
         tempprob = (float)hist[n] / (float)totpix;
         pix_freq[k].Freq = tempprob;
         huffcodes[k].Freq = tempprob;

         // Declaring the child of leaf
         // node as NULL pointer
         pix_freq[k].left = NULL;
         pix_freq[k].right = NULL;

         // initializing the code
         // word as end of line
         pix_freq[k].code[0] = '\0';
         k++;
      }
   }

}

void sortHist(huffcode* &huffcodes, int nodes)
{
     // Sorting the histogram
    int i, j;
    huffcode temphuff;

    // Sorting w.r.t probability
    // of occurrence
    for (i = 0; i < nodes; i++)
    {
        for (j = i + 1; j < nodes; j++)
        {
            if (huffcodes[i].Freq < huffcodes[j].Freq)
            {
                temphuff = huffcodes[i];
                huffcodes[i] = huffcodes[j];
                huffcodes[j] = temphuff;
            }
        }
    }
}

void BuildTree(pixfreq<25> *&pix_freq, huffcode* &huffcodes, int nodes)
{
    // Building Huffman Tree
    float sumprob;
    int sumpix,i;
    int n = 0, k = 0;
    int nextnode = nodes;

    // Since total number of
    // nodes in Huffman Tree
    // is 2*nodes-1
    while (n < nodes - 1)
    {

        // Adding the lowest two probabilities
        sumprob = huffcodes[nodes - n - 1].Freq + huffcodes[nodes - n - 2].Freq;
        sumpix = huffcodes[nodes - n - 1].intensity + huffcodes[nodes - n - 2].intensity;

        // Appending to the pix_freq Array
        pix_freq[nextnode].intensity = sumpix;
        pix_freq[nextnode].Freq = sumprob;
        pix_freq[nextnode].left = &pix_freq[huffcodes[nodes - n - 2].arrloc];
        pix_freq[nextnode].right = &pix_freq[huffcodes[nodes - n - 1].arrloc];
        pix_freq[nextnode].code[0] = '\0';
        i = 0;

        // Sorting and Updating the
        // huffcodes array simultaneously
        // New position of the combined node
        while (sumprob <= huffcodes[i].Freq)
            i++;

        // Inserting the new node
        // in the huffcodes array
        for (k = nodes; k >= 0; k--)
        {
            if (k == i)
            {
                huffcodes[k].intensity = sumpix;
                huffcodes[k].Freq = sumprob;
                huffcodes[k].arrloc = nextnode;
            }
            else if (k > i)

            // Shifting the nodes below
            // the new node by 1
            // For inserting the new node
            // at the updated position k
            huffcodes[k] = huffcodes[k - 1];

        }
        n += 1;
        nextnode += 1;
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
    // Assigning Code through
    // backtracking
    int i;
    char left = '0';
    char right = '1';
    for (i = totalnodes - 1; i >= nodes; i--)
    {
        if (pix_freq[i].left != NULL)
        {
            stradd(pix_freq[i].left->code, pix_freq[i].code, left);
        }
        if (pix_freq[i].right != NULL)
        {
            stradd(pix_freq[i].right->code, pix_freq[i].code, right);
        }
    }
}

#endif /* COMPRESS_HELPER_H_ */