 #include <stdio.h>
int main()
{
int celestini[6] = {6,5,4,3,2,1};
int *ptr = (int*)(&celestini+1);
printf("%d %d", *(celestini+1), *(ptr-1));
return 0;
}

