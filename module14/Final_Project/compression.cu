// C Code for
// Image Compression
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Utilities.h"
#include <string>
#include <fstream>

#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "ImageIO.h"
#include "Exceptions.h"
#include <npp.h>


using namespace std;

// function to calculate word length
int codelen(char* code)
{
   int l = 0;
   while (*(code + l) != '\0')
      l++;
   return l;
}

// function to concatenate the words
void strconcat(char* str, char* parentcode, char add)
{
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


void LoadImagePGM(npp::ImageCPU_8u_C1 &hostImage)
{
    cout<<"Start...\n";
    string name = "Lena.pgm";

    ifstream inputfile(name.data(), std::ifstream::in);

    if (inputfile.good())
    {
        cout << "assignmentNPP opened: <" << name.data() << "> successfully!" << endl;
        inputfile.close();
    }
    else
    {
        cout << "assignmentNPP unable to open: <" << name.data() << ">" << endl;
        inputfile.close();
        exit(EXIT_FAILURE);
    }

   // declare a host image object for an 8-bit grayscale image
    

    // load gray-scale image from disk
    npp::loadImage(name, hostImage);

}


// Driver code
int main()
{
   int i, j;
  npp::ImageCPU_8u_C1 hostImage; 
  LoadImagePGM(hostImage);
   

   const int height = hostImage.height();
   const int width = hostImage.width();
   int image[height][width];

    for (i = 0; i < height; i++)
    {
      for (j = 0; j < width; j++)
      {
         image[i][j] = static_cast<int>(*(hostImage.data(i,j)));
      }
    }
    

   // Finding the probability
   // of occurrence
   
   int hist[256];
   for (i = 0; i < 256; i++)
      hist[i] = 0;
   for (i = 0; i < height; i++)
      for (j = 0; j < width; j++)
      {
         if(image[i][j]>=256)
            cout<<"Este es el problema ="<<image[i][j]<<endl;
         hist[image[i][j]] += 1;
      }


   
   // Finding number of
   // non-zero occurrences
   int nodes = 0;
   for (i = 0; i < 256; i++)
      if (hist[i] != 0)
         nodes += 1;

                  


   // Calculating minimum probability
   float p = 1.0, ptemp;
   for (i = 0; i < 256; i++)
   {
      ptemp = (hist[i] / (float)(height * width));
      if (ptemp > 0 && ptemp <= p)
         p = ptemp;
   }
   
   // Calculating max length
   // of code word
   i = 0;
   while ((1 / p) > fib(i))
   {
      i++;
   }
   int maxcodelen = i - 3;
   cout<<"maxcodelen = "<<maxcodelen<<endl;

   // Declaring structs
   //struct pixfreq<maxcodelen> *pix_freq; it should be this!!
   pixfreq<25> *pix_freq;
   huffcode* huffcodes;
   int totalnodes = 2 * nodes - 1;
   pix_freq = (pixfreq<25>*)malloc(sizeof(pixfreq<25>) * totalnodes);
   huffcodes = (struct huffcode*)malloc(sizeof(struct huffcode) * nodes);
   
   // Initializing
   j = 0;
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

   // Sorting the histogram
   struct huffcode temphuff;

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

   // Building Huffman Tree
   float sumprob;
   int sumpix;
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
cout<<"bien aqui<<"<<endl;
   // Assigning Code through
   // backtracking
   char left = '0';
   char right = '1';
   for (i = totalnodes - 1; i >= nodes; i--)
   {
      if (pix_freq[i].left != NULL)
         strconcat(pix_freq[i].left->code, pix_freq[i].code, left);
      if (pix_freq[i].right != NULL)
         strconcat(pix_freq[i].right->code, pix_freq[i].code, right);
   }

   // Encode the Image
   int pix_val;
   int l;

   // Writing the Huffman encoded
   // Image into a text file
   FILE* imagehuff = fopen("encoded_image.txt", "wb");
   for (i = 0; i < height; i++)
      for (j = 0; j < width; j++)
      {
         pix_val = image[i][j];
         for (l = 0; l < nodes; l++)
            if (pix_val == pix_freq[l].intensity)
               fprintf(imagehuff, "%s", pix_freq[l].code);
      }

   // Printing Huffman Codes
   printf("Huffmann Codes::\n\n");
   printf("pixel values -> Code\n\n");
   for (i = 0; i < nodes; i++) {
      if (snprintf(NULL, 0, "%d", pix_freq[i].intensity) == 2)
         printf("  %d    -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
      else
         printf(" %d  -> %s\n", pix_freq[i].intensity, pix_freq[i].code);
   }

   // Calculating Average Bit Length
   float avgbitnum = 0;
   for (i = 0; i < nodes; i++)
      avgbitnum += pix_freq[i].Freq * codelen(pix_freq[i].code);
   printf("Average number of bits:: %f", avgbitnum);

   //free(image);
}
