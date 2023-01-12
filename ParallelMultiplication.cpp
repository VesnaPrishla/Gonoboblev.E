#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#include "mpi.h"
#include <iostream>

int M = 2048;
int N = 1536;

class Matrix
{
public:
	int* _matrix = NULL;

public:
	int Rows = 0;
	int Cols = 0;


public:
	Matrix(int rows, int cols, bool empty = true)
	{
		Rows = rows;
		Cols = cols;
		_createMatrix(rows, cols, empty);
	}

	~Matrix()
	{
		if (_matrix != NULL) delete[] _matrix;
		_matrix = NULL;
	}
private:
	void _createMatrix(int rows, int cols, bool empty)
	{
		_matrix = new int[rows * cols];
		for (int row = 0; row < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				int initNumber = 0;
				if (empty == false) initNumber = 1 + std::rand() % 10;

				_matrix[(row * cols + col)] = initNumber;
			}
		}
	}

public:
	int GetValue(int row, int col)
	{
		return _matrix[(row * Cols + col)];
	}

	void SetValue(int row, int col, int value)
	{
		_matrix[(row * Cols + col)] = value;
	}
};






void GetRows(int numrowsMatrixA, int rank, int worldsize, int* indexbegin, int* countrows)
{
	int d = numrowsMatrixA % worldsize;
	int wr = (numrowsMatrixA - d) / worldsize;

	int bgn = 0;
	if (rank > 0) bgn = wr * rank + d;
	int rows = wr + d;
	if (rank > 0) rows = wr;

	*indexbegin = bgn;
	*countrows = rows;
}



void MultiplicationMatrixSegments(int* rowsMatrixA, int* outbuffer, int countrowsMatrixA, Matrix* B, Matrix* matrixC)
{
	for (int mArow = 0; mArow < countrowsMatrixA; mArow++)
	{
		for (int Bcol = 0; Bcol < M; Bcol++)
		{
			int C = 0;
			for (int BrowAcol = 0; BrowAcol < N; BrowAcol++)
			{
				int t = mArow * N + BrowAcol;
				C += rowsMatrixA[t] * (*B).GetValue(BrowAcol, Bcol);
			}
			if (outbuffer != NULL) outbuffer[mArow * M + Bcol] = C;
			if (matrixC != NULL) (*matrixC).SetValue(mArow, Bcol, C);
		}
	}
}




void  MainProcessor(int world_rank, int world_size)
{


	std::srand(std::time(nullptr));
	Matrix matrixA = Matrix(M, N, false);
	Matrix matrixB = Matrix(N, M, false);
	Matrix matrixC(M, M);

	for (int i = 1; i < world_size; i++)
	{
		int index = 0, countrows = 0;
		GetRows(M, i, world_size, &index, &countrows);

		int countElements = countrows * N;
		int offsetBuffer = index * N;
		int error1 = MPI_Send(matrixA._matrix + offsetBuffer, countElements, MPI_INT, i, 0, MPI_COMM_WORLD);
		int error = MPI_Send(matrixB._matrix, N * M, MPI_INT, i, 0, MPI_COMM_WORLD);
	}






	double startMultiply = MPI_Wtime();

	int index = 0, countrows = 0;
	GetRows(M, world_rank, world_size, &index, &countrows);

	MultiplicationMatrixSegments(matrixA._matrix + index * N, NULL, countrows, &matrixB, &matrixC);

	double endMultiply = MPI_Wtime();
	double timeMultiplication = endMultiply - startMultiply;
	std::cout << "Time of multiplication: " << timeMultiplication << " sec.\n";


	for (int i = 1; i < world_size; i++)
	{
		int index = 0, countrows = 0;
		int rank = i;
		int size_comm = world_size;

		GetRows(M, rank, size_comm, &index, &countrows);

		int* recvRowsMatrixC = new int[countrows * M];
		memset(recvRowsMatrixC, 0, countrows * M);

		int errorRecv = MPI_Recv(recvRowsMatrixC, countrows * M, MPI_INT, i, 0, MPI_COMM_WORLD,
			MPI_STATUS_IGNORE);
		memcpy(matrixC._matrix + (index * M), recvRowsMatrixC, countrows * M * sizeof(int));

		delete[] recvRowsMatrixC;
	}
}



void SubProcessors(int world_rank, int world_size)
{

	Matrix recvB = Matrix(N, M);

	int indexBeginRows = 0;
	int countrowsMatrixA = 0;
	GetRows(M, world_rank, world_size, &indexBeginRows, &countrowsMatrixA);
	int* recvRowsMatrixA = new int[countrowsMatrixA * N];

	int error1 = MPI_Recv(recvRowsMatrixA, countrowsMatrixA * N, MPI_INT, 0, 0, MPI_COMM_WORLD,
		MPI_STATUS_IGNORE);


	int error = MPI_Recv(recvB._matrix, N * M, MPI_INT, 0, 0, MPI_COMM_WORLD,
		MPI_STATUS_IGNORE);

	int* outRowsMatrixC = new int[countrowsMatrixA * M];

	MultiplicationMatrixSegments(recvRowsMatrixA, outRowsMatrixC, countrowsMatrixA, &recvB, NULL);
	delete[] recvRowsMatrixA;

	int errorSend = MPI_Send(outRowsMatrixC, countrowsMatrixA * M, MPI_INT, 0, 0, MPI_COMM_WORLD);
	delete[] outRowsMatrixC;
}




int main(int argc, char* argv[])
{

	MPI_Init(NULL, NULL);
	int world_size, world_rank;

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if (world_rank == 0)
		MainProcessor(world_rank, world_size);
	else
		SubProcessors(world_rank, world_size);


	MPI_Finalize();
	_CrtDumpMemoryLeaks();

}