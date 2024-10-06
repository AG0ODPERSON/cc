import numpy as np  
  
def circular_convolution(x, h, N):  
    """  
    计算两个序列的指定点数的圆周卷积  
  
    参数:  
    x -- 第一个输入序列（numpy数组）  
    h -- 第二个输入序列（numpy数组）  
    N -- 圆周卷积的点数（整数）  
  
    返回:  
    y -- 圆周卷积结果（numpy数组），长度为N  
    """  
    # 使用零填充将两个序列扩展到长度N  
    x_padded = np.pad(x, (0, N - len(x)), mode='constant', constant_values=0)  
    h_padded = np.pad(h, (0, N - len(h)), mode='constant', constant_values=0)  
  
    # 计算离散傅里叶变换  
    X = np.fft.fft(x_padded)  
    H = np.fft.fft(h_padded)  
  
    # 计算频域中的乘积  
    Y = X * H  
  
    # 计算逆离散傅里叶变换得到圆周卷积结果  
    y = np.fft.ifft(Y)  
  
    # 由于ifft返回复数结果，但圆周卷积对于实数输入应产生实数输出（在数值误差范围内）  
    # 这里我们取实部，但通常对于实数输入，虚部应该非常小  
    y = np.real(y)  
  
    # 返回前N个点作为结果  
    return y[:N]  
  
# 示例  
if __name__ == "__main__":  
    # 输入序列  
    x = np.array([-1899,-633,-1266,0,-2406,-802,-1604,0])  
    h = np.array([-633,0,0,0,802,0,0,0])  
  
    # 指定圆周卷积的点数  
    N = int(input("请输入圆周卷积的点数（N）: "))  
  
    # 确保N足够大以容纳所有非零值  4
    # （这不是严格必要的，因为用户可能已经知道他们想要的结果长度）   
    # 计算圆周卷积  
    y = circular_convolution(x, h, N)  
  
    # 打印结果  
    print("圆周卷积结果:", y)