// C Code for
// Image Compression
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "Utilities.h"

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



void readBMPFILE(int &width, int &height, int* image)
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
         image = (int*)malloc(height * width * sizeof(int));

         // Reading the BMP File
         // into Image Array
         //cout<<"height = "<<height<<endl;
         //cout<<"width = "<<width<<endl;
         //cout<<"height * width = "<<height * width<<endl;

         for (i = 0; i < height; i++)
         {
            for (j = 0; j < width; j++)
            {
               fread(&temp, 3, 1, image_file);

               // the Image is a
               // 24-bit BMP Image
               temp = temp & 0x0000FF;
               image[index(i,j,height)] = temp;
               //cout<<"index("<<i<<")("<<j<<")("<<height<<") = "<<index(i,j,height)<<endl;
               if(image[index(i,j,height)]>256)
                  cout<<"image["<<i<<"]["<<j<<"] = "<<image[index(i,j,height)]<<" ";
            }
         }
      }

}

// Driver code
int main()
{
   int i, j;
   int width, height;
   int* image;

   readBMPFILE(width, height, image);

   
   // Finding the probability
   // of occurrence
   int hist[256];
   for (i = 0; i < 256; i++)
      hist[i] = 0;
   for (i = 0; i < height; i++)
      for (j = 0; j < width; j++)
         hist[image[index(i,j,height)]] += 1;
   cout<<"Entre aqui"<<endl;     
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
   //int maxcodelen = i - 3;

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
         pix_val = image[index(i,j,height)];
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

   free(image);
}
