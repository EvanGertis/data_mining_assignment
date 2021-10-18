lambda_1=36.2211
lambda_2=6.98907
lambda_3=1.58981

varianceOfv1 = lambda_1/(lambda_1+lambda_2+lambda_3)
varianceOfv2 = lambda_2/(lambda_1+lambda_2+lambda_3)
varianceOfv3 = lambda_3/(lambda_1+lambda_2+lambda_3)


print(f' variance of v1 : {varianceOfv1}')
print(f' variance of v2 : {varianceOfv2}')
print(f' variance of v3 : {varianceOfv3}')