/*---------------------------------------------------------------------------

 K-NN classifier.

 

 
Synopsis:  
  
[ytest , proba] = knnpredic(Xtrain , ytrain , Xtest , [k]);

Inputs:
------
	
Xtrain           Training data [d x Ntrain]

ytrain           Labels of training data [1 x Ntrain]
	
Xtest            Data to be classified [d x Ntest] 
	
k                Number of neighbours (default k = 3)
	
Output:
-------

ytest            Estimated labels of testing data [1 x Ntest] 


proba            Estimated Pr(w=j|x), j=1,...,m where m is the number of classes. (m x Ntest)
	 
	   
Example 1
---------		 

d                     = 3;
Ntrain                = 100;
Ntest                 = 100;
k                     = 4;
		   
Xtrain                = randn(d , Ntrain);
ytrain                = double(rand(1 , Ntrain) > 0.5);
Xtest                 = randn(d , Ntest);
			 
[ytest_est , proba]   = knnpredic(Xtrain , ytrain , Xtest , k);


Example 2
---------		 

close all hidden
d                   = 2;
Ntrain              = 100;
k                   = 4;

vect                = (-5:0.25:5);
N                   = length(vect); 
 
Xtrain              = randn(d , Ntrain);
ytrain              = double(rand(1 , Ntrain) > 0.5);
ind0                = (ytrain==0);
ind1                = (ytrain==1);

[X , Y]             = meshgrid(vect);
Xtest               = [X(:)' ; Y(:)'];



[ytest_est , proba] = knnpredic(Xtrain , ytrain , Xtest , k);

figure(1)
pcolor(vect , vect , reshape(ytest_est , N , N))
hold on
plot(Xtrain(1 , ind0) , Xtrain(2 ,ind0) , 'y+' , Xtrain(1 , ind1) , Xtrain(2 , ind1) , 'm*')
hold off

figure(2)
pcolor(vect , vect , reshape(proba(1 , :) , N , N))
hold on
plot(Xtrain(1 , ind0) , Xtrain(2 ,ind0) , 'y+' , Xtrain(1 , ind1) , Xtrain(2 , ind1) , 'm*')
hold off



Example 3
---------		 

close all hidden
d                 = 2;
Ntrain            = 2000;
k                 = 5;

vect              = (-5:0.25:5);
N                 = length(vect); 
m                 = 2;
M0                = [-0 ; -0];
R0                = [1 0 ; 0 1];
M1                = [1 ; 1];
R1                = [0.5 0.1 ; 0.2 1];

 
Xtrain            = [M0(: , ones(1 , Ntrain/2)) + chol(R0)'*randn(d , Ntrain/2) , M1(: , ones(1 , Ntrain/2)) + chol(R1)'*randn(d , Ntrain/2)]; 
ytrain            = [zeros(1 , Ntrain/2) , ones(1 , Ntrain/2)];
ind0              = (ytrain==0);
ind1              = (ytrain==1);

[X , Y]           = meshgrid(vect);
Xtest             = [X(:)' ; Y(:)'];



[ytest_est , proba]   = knnpredic(Xtrain , ytrain , Xtest , k);

figure(1)
pcolor(vect , vect , reshape(ytest_est , N , N))
hold on
plot(Xtrain(1 , ind0) , Xtrain(2 ,ind0) , 'y+' , Xtrain(1 , ind1) , Xtrain(2 , ind1) , 'm*')
hold off

figure(2)
pcolor(vect , vect , reshape(proba(1 , :) , N , N))
hold on
plot(Xtrain(1 , ind0) , Xtrain(2 ,ind0) , 'y+' , Xtrain(1 , ind1) , Xtrain(2 , ind1) , 'm*')
hold off


Example 4
---------

load wine

N          = size(X , 2);
n1         = round(0.7*N);
n2         = N - n1;
nbite      = 1000;
Perf       = zeros(1 , nbite);
k          = 3;

for i=1:nbite

 ind        = randperm(length(y));

 ind1       = ind(1:n1);
 ind2       = ind(n1+1:N);

 Xtrain     = X(: , ind1);
 ytrain     = y(ind1);
 Xtest      = X(: , ind2);

 ytest      = y(ind2);
 ytest_est  = knnpredic(Xtrain , ytrain , Xtest , k );

 Perf(i)    = knnpredic(ytest == ytest_est)/n2;

end
disp(mean(Perf))

 Author : Sébastien PARIS : sebastien.paris@lsis.org
 -------  Date : 15/03/2007



mex -f mexopts_intelamd.bat -output knnpredic.dll knnpredic.c


				 
-------------------------------------------------------------------- 

*/

#include <math.h>
#include <limits.h>
#include "mex.h"

#define MAX_INF INT_MAX

#define MAX(A,B)   (((A) > (B)) ? (A) : (B) )
#define MIN(A,B)   (((A) < (B)) ? (A) : (B) ) 




void qs( double * , int , int  ); 



void mexFunction( int nlhs, mxArray *plhs[] , int nrhs, const mxArray*prhs[] )
{
	double *Xtest, *Xtrain , *ytrain;
	double *ytest , *proba ;
	
	double *dist;
	int    *max_labels;
	
	int  Ntest, Ntrain, k = 3 , i, j, l, d , id , jd , im;
	
	double temp , adist, max_dist;
	
	int  max_inx, best_count, best_label, count , ind , m=0;

    double  currentlabel , sumdistmini;


	double *ytrainsorted , *labels , *distmini;

	

	if( nrhs < 3) 
	{
		mexErrMsgTxt("Incorrect number of input arguments.");
	}
	

	/* -- gets input arguments --------------------------- */
	
	Xtrain   = mxGetPr(prhs[0]);
	
	d        = mxGetM(prhs[0]);              /* data dimension */
	Ntrain   = mxGetN(prhs[0]);              /* number of data */
	

	ytrain   = mxGetPr(prhs[1]);

	
	Xtest    = mxGetPr(prhs[2]);

	Ntest    = mxGetN(prhs[2]);              /* number of data */
	
	if( d != mxGetM( prhs[2] )) 

	{
		
		mexErrMsgTxt("Dimension of training and testing data differs.");
	}
	
	if(nrhs>3)
	{
		
		k          = (int) mxGetScalar(prhs[3]);    
		
	}




	/* Determine unique Labels */
	
	ytrainsorted  = mxMalloc(Ntrain*sizeof(double));
	
	
	for ( i = 0 ; i < Ntrain; i++ ) 
	{
		
		ytrainsorted[i] = ytrain[i];
		
	}
	
	
	qs( ytrainsorted , 0 , Ntrain - 1 );
	
	
	labels       = mxMalloc(sizeof(double)); 
	
	labels[m]    = ytrainsorted[0];
	
	currentlabel = labels[0];
	
	for (i = 0 ; i < Ntrain ; i++) 
	{ 
		if (currentlabel != ytrainsorted[i]) 
		{ 
			labels       = (double *)mxRealloc(labels , (m+2)*sizeof(double)); 
			
			labels[++m]  = ytrainsorted[i]; 
			
			currentlabel = ytrainsorted[i];
			
		} 
	} 
	
	m++; 



	
	/*  output labels*/
	
	plhs[0]    = mxCreateDoubleMatrix(1 , Ntest , mxREAL);

	ytest      = mxGetPr(plhs[0] );


	plhs[1]    = mxCreateDoubleMatrix(m , Ntest , mxREAL);

	proba      = mxGetPr(plhs[1] );
	
	/*--------------------------*/

	
	dist       = mxMalloc(Ntrain*sizeof(double));
	
	max_labels = mxMalloc(k*sizeof(int));

	distmini   = mxMalloc(m*sizeof(double));

	
	
	for( i = 0 ; i < Ntest; i++ ) 
	{
		id      = i*d;

		im      = i*m;

		for (j = 0 ; j < m ; j++)

		{

			distmini[j] = MAX_INF;

		}
		
		for( j = 0 ; j < Ntrain ; j++ ) 
		{
			
			jd    = j*d;
			
			adist = 0.0;
			
			for( l = 0 ; l < d ; l++ ) 
			{
				temp   = (Xtest[l + id] - Xtrain[l + jd]);
				
				adist += temp*temp; 
			}
			
			dist[j]  = sqrt(adist);
			
			for (l = 0 ; l < m ; l++)
			{
				
				if (ytrain[j] == labels[l])
					
				{
					
					ind = l;
					
				}
			}

			if (dist[j] < distmini[ind])
			{

				distmini[ind] = dist[j];

			}

		}
		
		sumdistmini  = 0.0;


		for (j = 0 ; j < m ; j++)

		{

			sumdistmini += distmini[j];

		}

		sumdistmini  = 1.0/sumdistmini;

		for( l = 0 ; l < k ; l++) 
		{
			
			max_dist = MAX_INF;
			
			for( j = 0 ; j < Ntrain ; j++ ) 
			{
				if( max_dist > dist[j] ) 
				{
					max_inx  = j;
					
					max_dist = dist[j];
				}
			}
			
			dist[ max_inx ] = MAX_INF;
			
			max_labels[l]   = (int) ytrain[max_inx];
		}
		
		best_count = 0;
		
		for( l = 0 ; l < k; l++) 
		{
			count = 0;

			ind   = max_labels[l];
			
			for( j = 0 ; j < k; j++) 
			{
				if( ind == max_labels[j] ) 
				{
					
					count++;
				}
			}
			if( count > best_count ) 
			{
				
				best_count = count;
				
				best_label = max_labels[l]; 
			}
		}    
		
		ytest[i] = best_label;


		for (l = 0 ; l < m ; l++)
		{


			proba[l + im] = 1.0 - (m - 1.0)*distmini[l]*sumdistmini;

		}

		
	}
  
  mxFree(dist);  

  mxFree(max_labels);

  mxFree(labels);
	
  mxFree(ytrainsorted);

  mxFree(distmini);

}




/*-------------------------------------------------------------------------------------------------------------- */


void qs( double *array , int left , int right ) 
{
	
	double pivot;	// pivot element.
	
	int holex;	// hole index.
	
	int i;
	
	holex          = left + ( right - left )/2;

	pivot          = array[ holex ];		     // get pivot from middle of array.
	
	array[holex]   = array[ left ];              // move "hole" to beginning of
	
	holex          = left;			             // range we are sorting.
	
	for ( i = left + 1 ; i <= right ; i++ ) 
	{
		if ( array[ i ] <= pivot ) 
		{
			array[ holex ] = array[ i ];

			array[ i ]     = array[ ++holex ];
		}
	}
	
	if ( holex - left > 1 ) 
	{
	
		qs( array, left, holex - 1 );
	
	}
	if ( right - holex > 1 ) 
	{
	
		qs( array, holex + 1, right );
	
	}
	
	array[ holex ] = pivot;
	
}


/* ----------------------------------------------------------------------------- */
