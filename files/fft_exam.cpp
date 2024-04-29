


... FFT


inline void FFT(short int dir,long m,double *x,double *y)
{
    long n,i,i1,j,k,i2,l,l1,l2;
    double c1,c2,tx,ty,t1,t2,u1,u2,z;

    n = 1;
    for (i=0;i<m;i++)
        n *= 2;


    i2 = n >> 1;
    j = 0;


    for (i=0;i<n-1;i++) {
        if (i < j) {
            tx = x[i];
            ty = y[i];
            x[i] = x[j];
            y[i] = y[j];
            x[j] = tx;
            y[j] = ty;
        }
        k = i2;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }




    c1 = -1.0;
    c2 = 0.0;
    l2 = 1;
    for (l=0;l<m;l++) {
        l1 = l2;
        l2 <<= 1;
        u1 = 1.0;
        u2 = 0.0;
        for (j=0;j<l1;j++) {
            for (i=j;i<n;i+=l2) {
                i1 = i + l1;
                t1 = u1 * x[i1] - u2 * y[i1];
                t2 = u1 * y[i1] + u2 * x[i1];
                x[i1] = x[i] - t1;
                y[i1] = y[i] - t2;
                x[i] += t1;
                y[i] += t2;
            }
            z =  u1 * c1 - u2 * c2;
            u2 = u1 * c2 + u2 * c1;
            u1 = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (dir == 1)
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    if (dir == 1) {
        for (i=0;i<n;i++) {
            x[i] /= n;
            y[i] /= n;
        }
    }
}












main()
{

...
double nd = log2((double)data.size());

   int N =  exp2 ((int)nd);

   double Xi[N];
   double Yi[N];

   for(size_t i=0;i<N;i++)
   {
       Xi[i]=data[i];
       Yi[i]=0.0;

   }

   FFT(1,(int)nd,Xi,Yi);
   ...
   }







.....    The 1D convolutional neural network

https://github.com/harryjdavies/Python1D_CNNs/blob/master/CNN1D_keras_activity.py
