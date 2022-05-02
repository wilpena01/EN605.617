#ifndef COMPRESS_HELPER_H_
#define COMPRESS_HELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Utilities.h"
#include <string>
#include <fstream>

using namespace std;

void strconcat(char* str, char* parentcode, char add)
{
    // function to concatenate the words
   int i = 0;
   while (*(parentcode + i) != '\0')
   {
      *(str + i) = *(parentcode + i);
      i++;
   }
   if (add != '2')
   {
      str[i] = add;
      str[i + 1] = '\0';
   }
   else
      str[i] = '\0';
}

void readBMPFILE(int &width, int &height, int** &image)
{
      int i, j;
      char filename[] = "Lena.bmp";
      int offset, bpp = 0;
      long bmpsize = 0, bmpdataoff = 0;
      int temp = 0;
   // Reading the BMP File
      FILE* image_file;

      image_file = fopen(filename, "rb");
      if (image_file == NULL)
      {
         printf("Error Opening File!!");
         exit(1);
      }
      else
      {

         // Set file position of the
         // stream to the beginning
         // Contains file signature
         // or ID "BM"
         offset = 0;

         // Set offset to 2, which
         // contains size of BMP File
         offset = 2;

         fseek(image_file, offset, SEEK_SET);

         // Getting size of BMP File
         fread(&bmpsize, 4, 1, image_file);

         // Getting offset where the
         // pixel array starts
         // Since the information is
         // at offset 10 from the start,
         // as given in BMP Header
         offset = 10;

         fseek(image_file, offset, SEEK_SET);

         // Bitmap data offset
         fread(&bmpdataoff, 4, 1, image_file);

         // Getting height and width of the image
         // Width is stored at offset 18 and
         // height at offset 22, each of 4 bytes
         fseek(image_file, 18, SEEK_SET);

         fread(&width, 4, 1, image_file);

         fread(&height, 4, 1, image_file);

         // Number of bits per pixel
         fseek(image_file, 2, SEEK_CUR);

         fread(&bpp, 2, 1, image_file);

         // Setting offset to start of pixel data
         fseek(image_file, bmpdataoff, SEEK_SET);

         // Creating Image array
         image = (int**)malloc(height * sizeof(int*));

         for (i = 0; i < height; i++)
         {
            image[i] = (int*)malloc(width * sizeof(int*));
         }
         // Reading the BMP File
         // into Image Array
         for (i = 0; i < height; i++)
         {
            for (j = 0; j < width; j++)
            {
               fread(&temp, 3, 1, image_file);

               // the Image is a
               // 24-bit BMP Image
               temp = temp & 0x0000FF;
               image[i][j] = static_cast<int>(temp);
            }
         }
      }
      fclose(image_file);
}

void ocurrence(int* hist, int** image, int width, int height)
{
    // Finding the probability
    // of occurrence
    int i,j;
   
    for (i = 0; i < 256; i++)
        hist[i] = 0;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            hist[image[i][j]] += 1;
        }
    }
}

void nonZero_ocurrence(int* hist, int &node)
{
    // Finding number of
    // non-zero occurrences
    node=0;
    for (int i = 0; i < 256; i++)
      if (hist[i] != 0)
         node += 1;
}

void minProp(float &p, int* hist, int width, int height)
{
    // Calculating minimum probability
    float ptemp;
    p = 1.0;
    for (int i = 0; i < 256; i++)
    {
        ptemp = (hist[i] / (float)(height * width));
        if (ptemp > 0 && ptemp <= p)
            p = ptemp;
    }
}

int MaxLength(float p)
{
    // Calculating max length
    // of code word
    int i = 0;
    while ((1 / p) > fib(i))
    {
        i++;
    }

    return i;
}

void InitStruct(pixfreq<25> *pix_freq, huffcode* huffcodes, 
                int* hist, int height, int width)
{
     // Initializing
   int i; int j=0;
   int totpix = height * width;
   float tempprob;
   for (i = 0; i < 256; i++)
   {
      if (hist[i] != 0)
      {

         // pixel intensity value
         huffcodes[j].intensity = i;
         pix_freq[j].intensity = i;

         // location of the node
         // in the pix_freq array
         huffcodes[j].arrloc = j;

         // probability of occurrence
         tempprob = (float)hist[i] / (float)totpix;
         pix_freq[j].Freq = tempprob;
         huffcodes[j].Freq = tempprob;

         // Declaring the child of leaf
         // node as NULL pointer
         pix_freq[j].left = NULL;
         pix_freq[j].right = NULL;

         // initializing the code
         // word as end of line
         pix_freq[j].code[0] = '\0';
         j++;
      }
   }

}

void sortHist(huffcode* huffcodes, int nodes)
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

void BuildTree(pixfreq<25> *pix_freq, huffcode* huffcodes, int nodes)
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

void AssignCode(pixfreq<25> *pix_freq, int nodes, int totalnodes)
{
    // Assigning Code through
    // backtracking
    int i;
    char left = '0';
    char right = '1';
    for (i = totalnodes - 1; i >= nodes; i--)
    {
        if (pix_freq[i].left != NULL)
            strconcat(pix_freq[i].left->code, pix_freq[i].code, left);
        if (pix_freq[i].right != NULL)
            strconcat(pix_freq[i].right->code, pix_freq[i].code, right);
        cout<<" i = "<<i<<endl;
    }
}

#endif /* COMPRESS_HELPER_H_ */