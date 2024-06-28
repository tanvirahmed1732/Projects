#include <stdio.h>
#include <string.h>
void addCharToBeginning(char str[], char ch) {
    int length = strlen(str);

    for (int i = length; i >= 0; i--) {
        str[i + 1] = str[i];
    }

    str[0] = ch;
}

void removeFirstChar(char str[]) {
    int length = strlen(str);
    for (int i = 0; i < length; i++) {
        str[i] = str[i + 1];
    }
}


int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int modInverse(int a, int m) {
    int m0 = m, t, q;
    int x0 = 0, x1 = 1;

    if (m == 1)
        return 0;

    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }

    if (x1 < 0)
        x1 += m0;

    return x1;
}

int encode(int k, int x, int b, int m) {
    return (k * x + b) % m;
}

int decode(int k, int y, int b, int m) {
    int a_inv = modInverse(k, m);
    return (a_inv * (y - b + m)) % m;  // +m to handle negative y-b
}

int main()
{
   int i, x, key,kr;
   char str[100],rk;

   printf("\nPlease enter your message: ");
   gets(str);

   printf("\nPlease choose following options:\n");
   printf("1 = Encrypt the message\n");
   printf("2 = Decrypt the message.\n");
   scanf("%d", &x);

   switch(x)
   {
   case 1:
       printf("Please enter the key:  \n");
       scanf("%d", &key);

      for(i = 0; (i < 100 && str[i] != '\0'); i++){
        int as = str[i];
        str[i] = encode(key, as, 3, 116);
      }

        rk = (char)(key+3);
       addCharToBeginning(str, rk);
      printf("\nEncrypted message: %s\n", str);
      break;

   case 2:
      kr = (int)(str[0])-3;
      for(i = 0; (i < 100 && str[i] != '\0'); i++){
        int as = str[i];
        str[i] = decode(kr, as, 3, 116);
      }
      removeFirstChar(str);
      printf("\nDecrypted message: %s\n", str);
      break;

   default:
      printf("\nWrong option\n");
   }
   return 0;
}
