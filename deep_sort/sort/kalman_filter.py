# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        """
        In [11]: np.eye(3*2,3*2)
        Out[11]: 
        array([[1., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 1.]])
        --------------------------------
        array([[1., 0., 0., 1., 0., 0.],
               [0., 1., 0., 0., 1., 0.],
               [0., 0., 1., 0., 0., 1.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 1., 0.],
               [0., 0., 0., 0., 0., 1.]])
        """
        #运动矩阵,采用常量速度模型,速度分量不会变,变的是(x,y,a,h),
        #所以矩阵才构造这上面这个样子
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        """
        In [7]: np.eye(3,3*2)
        Out[7]: 
        array([[1., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0.]])
        """
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement #xyah
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel] #按列连接两个矩阵

        #貌似是定义了8个状态变量初始化???
        #这么定义有什么依据?
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5, #是不是相当于0了?aspect ratio应该相对稳定得多
            10 * self._std_weight_velocity * measurement[3]
        ]
        # modified by bigz
        # std = [
        #     2 * self._std_weight_position * measurement[3],
        #     2 * self._std_weight_position * measurement[3],
        #     1e-5,
        #     1e-5, 
        #     10 * self._std_weight_velocity * measurement[3],
        #     10 * self._std_weight_velocity * measurement[3],
        #     1e-5, #是不是相当于0了?aspect ratio应该相对稳定得多
        #     1e-5 * self._std_weight_velocity * measurement[3],
        # ]
        """
        np.diag(x),当x是一个1维数组时,输出一个以一维数组为对角线元素的矩阵,x为2维矩阵时,输出矩阵的对角线元素;
        np.square(x),计算x中各元素的平方;
        """
        covariance = np.diag(np.square(std))
        return mean, covariance

    #先求先验估计值以及先验估计协方差矩阵
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        #这个不懂??
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        #作为对角元素,创建8x8的矩阵,矩阵Q
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) #shape,8x8

        mean = np.dot(self._motion_mat, mean)
        #T为转置,先验误差协方差这Pk- = A(Pk-1)A^T + Q,Q为过程误差协方差矩阵
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        #import ipdb
        #ipdb.set_trace()

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        像状态meaan有8个分量,而测量mean只有4个分量,所以这里弄了个project
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        #这个算法相当于是将速度分量变为了0?
        # import ipdb
        # ipdb.set_trace()
        mean = np.dot(self._update_mat, mean)
        #4x8,8x8,8x4
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    #根据先验估计值以及先验协方差,计算卡尔曼增益,校正预测值
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        """
        linalg是scipy提供的线性代数函数库
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        #卡尔曼增益
        #cho_solve,solve the linear equations Ax=b,given the Cholesky factorization A.
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        #projected_mean是啥?是H*X^k-
        innovation = measurement - projected_mean

        #后验估计值 = 先验估计值 + 卡尔曼增益*(测量值-先验估计值)??
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        #后验协方差矩阵 = Pk^- - Kk*H*Pk^-
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        #把一个对称正定表示成一个下三角矩阵L和其共轭转置矩阵的乘积,它要求矩阵的所有
        #特征值必须大于0,故分解的下三角的对角元也是大于零的.
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
