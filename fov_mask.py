
import numpy as np
import cv2

import torch
from cupy.cuda import function
from pynvrtc.compiler import Program
from collections import namedtuple

kernel = """

extern "C"

__global__ void img_fov(float* fov_angle, float* vector_x, float* vector_y, float* pointCenter, float* img_size, float* img)
{
	//1. 计算线程ID
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    float face_size_x =  tan(fov_angle[0]/2);
    float face_size_y =  tan(fov_angle[1]/2);

	//2. 输入输出的图片尺寸
    float sizeoutx = img_size[0];
    float sizeouty = img_size[1];

	if(x<sizeoutx && y<sizeouty)
	{

    	//1. 求ERP中某点(x,y)的经纬度
        float lonERP = x * 2 * 3.1415926 / sizeoutx - 3.1415926;
        float latERP = 3.1415926/2 - y * 3.1415926 / sizeouty;

		//2. 计算笛卡尔坐标系的坐标
        float pointERP[3];
        pointERP[0] = cos(latERP) * cos(lonERP);
	    pointERP[1] = cos(latERP) * sin(lonERP);
        pointERP[2] = sin(latERP);

		//3. 求该点与球心的直线 与 视点与球心的直线的夹角 cos(theta)
        float angle = pointERP[0] * pointCenter[0] + pointERP[1] * pointCenter[1] + pointERP[2] * pointCenter[2];

		//4. 超过半个平面
        if (angle < 0)
            return;

		//5. 求该店在且片面上的坐标
        pointERP[0] /= angle;
        pointERP[1] /= angle;
        pointERP[2] /= angle;

		//6. 视点与该点的连线
        pointERP[0] -= pointCenter[0];
        pointERP[1] -= pointCenter[1];
        pointERP[2] -= pointCenter[2];

		//7. 求在切平面下x,y轴的投影

        float plane_x = pointERP[0] * vector_x[0] + pointERP[1] * vector_x[1] + pointERP[2] * vector_x[2];
        float plane_y = pointERP[0] * vector_y[0] + pointERP[1] * vector_y[1] + pointERP[2] * vector_y[2];

		//8.判断是否在切平面内
        if (plane_x <=  face_size_x && plane_x >= -face_size_x)
            plane_x += face_size_x;
        else
            plane_x = -1;

        if (plane_y <=  face_size_y && plane_y >= -face_size_y)
            plane_y += face_size_y;
        else
            plane_y = -1;

        if(plane_x < 0 || plane_y <0)
            return;
        else
        {
            img[int((y* sizeoutx  + x) +0)] = 255.0;     
        }
	}

}
"""

program = Program(kernel, 'img_fov.cu')
ptx = program.compile()

m = function.Module()

m.load(bytes(ptx.encode()))

def generate_fov_mask(viewport_lat, viewport_lon, img_in_Width, img_in_Height):

	# field of vision #

    fov_angle = np.array([120 * np.pi / 180, 120 * np.pi / 180]).astype(np.float32)

    img_size = np.array([img_in_Width, img_in_Height]).astype(np.float32) # WxH

    # basic vector
    vector_x = np.array([-np.sin(viewport_lon), np.cos(viewport_lon), 0]).astype(np.float32).reshape(3, 1)

    vector_y = np.array([np.sin(viewport_lat) * np.cos(viewport_lon),
                                 np.sin(viewport_lat) * np.sin(viewport_lon),
                                 - np.cos(viewport_lat)]).astype(np.float32).reshape(3, 1)
    pos_c= np.array([np.cos(viewport_lat) * np.cos(viewport_lon),
                                    np.cos(viewport_lat) * np.sin(viewport_lon),
                                  np.sin(viewport_lat)]).astype(np.float32).reshape(3, 1)

	# 2. transfer to cuda #
    vector_x = torch.from_numpy(vector_x).float().cuda()

    vector_y =torch.from_numpy(vector_y).float().cuda()

    fov_angle  = torch.from_numpy(fov_angle).float().cuda()

    img_size = torch.from_numpy(img_size).float().cuda()

    img = torch.zeros((img_in_Height,img_in_Width)).float().cuda()

    pos_c = torch.from_numpy(pos_c).float().cuda()


    f = m.get_function('img_fov')

    Stream = namedtuple('Stream', ['ptr'])
	
    s = Stream(ptr=torch.cuda.current_stream().cuda_stream)

    f(grid=((img_in_Width-32)//32+1,(img_in_Height-32)//32+1,1), block=(32,32,1), args=[fov_angle.data_ptr(), vector_x.data_ptr(), vector_y.data_ptr(), pos_c.data_ptr(),  img_size.data_ptr(), img.data_ptr()], stream=s)

    return img



if __name__ == "__main__":

	a = generate_fov_mask(50*np.pi/180, 0, 3840, 1920)

	print(a.shape)

	cv2.imwrite('a.png',a.cpu().numpy())




