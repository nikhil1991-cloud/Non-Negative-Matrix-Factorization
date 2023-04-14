# Non-Negative-Matrix-Factorization
Performs NNMF using Multiplicative Updates

This code does non-negative matrix factorization of an M by N non negative matrix Z into two non-negative matrices W (shape M by R) and H (shape R by N) such that 
Z = WH using Multiplicative Updates. The details of the method are mentioned in the function description. The basis R of the matrix factors, W and H, follows the condition that R < min(M,N). The algorithm adopted is mentioned in section 4 of this paper: https://papers.nips.cc/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf.

An application of this method can be applied to mathematical models in physics where the model parameters strictly non-negative.
