#include "nac.h"

using namespace std;
using namespace cv;

nac::nac(){
    // assign parameter values
    tde = 0;
    updateFlag = false;
}

nac::~nac(){
}

/*********************************************************/
/******************network functions**********************/
/*********************************************************/

// initialize network
void nac::init(int _nvI, int _nvY, int _npI, int _npY, float _alpha, float _beta, float _gamma, float _wr){
    // assign parameter values
    // value network
    nvI = _nvI; nvY = _nvY;
    // policy network
    npI = _npI; npY = _npY;
    // learning rates 
    alpha = _alpha; beta = _beta; gamma = _gamma; wr = _wr;
    // allocate space for weights, input alread biased...
    // value network  
    vy = Mat::zeros(nvY, 1, CV_32F); 
    vyp = Mat::zeros(nvY, 1, CV_32F);
    vw.create(nvY, nvI, CV_32F);
    // policy network  
    py = Mat::zeros(npY, 1, CV_32F);
    pd = Mat::zeros(npY, 1, CV_32F);
    pw.create(npY, npI, CV_32F);
    cov = Mat::zeros(npY, npY, CV_32F);
    invCov = Mat::zeros(npY, npY, CV_32F);
    action = Mat::zeros(npY, 1, CV_32F);
    // log gradients
    dlpw = Mat::zeros(npY, npI, CV_32F);
    // natural gradients
    ngpw = Mat::zeros(npY, npI, CV_32F);
    dngpw = Mat::zeros(npY, npI, CV_32F);
    gIw = Mat::eye(npY * npI, npY * npI, CV_32F);
    psiW = Mat::zeros(npY * npI, 1, CV_32F);
    gPsiW = Mat::zeros(npY * npI, 1, CV_32F); psiGPsiW = Mat::zeros(1, 1, CV_32F);
    dpw = Mat::zeros(npY, npI, CV_32F);
    // randomize the weights
    randu(vw, -wr, wr); randu(pw, -wr, wr);
    alpha_ng = Mat::ones(npY, npI, CV_32F);
}

// update network
void nac::update(const Mat& ps, const Mat& vs, float reward){
    // value network forward propagation
    vy = vw * vs;
    // update: value network back prop, policy network natural gradient
    if(updateFlag){
        // value network update
        tde = reward + gamma * vy.at<float>(0) - vyp.at<float>(0); // td error
        dvw = alpha * tde * vs.t(); // value network update
        vw = vw + dvw; // perform update
        // policy network update
        pd = invCov * (action - py); // policy delta
        dlpw = pd * ps.t(); // delta * ps'
        dngpw = dlpw * dlpw.t() * ngpw - tde * dlpw;
        ngpw = ngpw - dngpw.mul(alpha_ng); // alpha_ng * dngpw; // natural gradient
        pw = pw + beta * ngpw; // update the weights of policy network tde * dlpw
    }
    // copy value output to previous value output
    vy.copyTo(vyp);
}

// update network using fisher information matrix
void nac::updateFisher(const Mat& ps, const Mat& vs, float reward, bool fisherFlag){
    // value network forward propagation
    vy = vv * vz;
    // update: value network back prop, policy network natural gradient
    if(fisherFlag && updateFlag){
        pd = invCov * (action - py); // all parameters are fixed... calculating delta
        dlpw = pd * ps.t(); // delta * ps'
        psiW = dlpw.reshape(1, npY * npI); // reshape normal gradient
        gPsiW = gIw * psiW; psiGPsiW = psiW.t() * gPsiW; // middle variables for inv fisher
        gIw = 1 / (1 - alpha) * (gIw - alpha * (gPsiW * gPsiW.t()) /  // inverse fisher 
                (1 - alpha + alpha * psiGPsiW.at<float>(0, 0)));
    }
    else if(updateFlag){
        // update average reward
        tde = reward + gamma * vy.at<float>(0) - vyp.at<float>(0); // td error
        dvw = alpha * tde * vs.t(); // value network update
        vw = vw + dvw; // update weights
        // policy network update
        pd = invCov * (action - py); // policy delta 
        dlpw = pd * ps.t(); // delta * x'
        psiW = dlpw.reshape(1, npY * npI); // reshape normal gradient
        ngpw = gIw * psiW; // natural gradient
        dpw = ngpw.reshape(1, npY); // delta weights
        pw = pw + beta * tde * dpw; // update the weights of policy network tde * dlpw
    }
    vy.copyTo(vyp); // copy new value to previous storages
}

// evaluate policy
void nac::evalPol(const Mat& ps){
    // policy network forward propagation
    py = pw * ps;
}


/*******************************************************************/
/**********************low level exploring functions****************/
/*******************************************************************/


// other functions
// set covariance matrix parameters...
void nac::setCov(float xr, float yr, float lxr, float sxr, float ar){
    // ranges are preprocessed and equal to 2 standard deviation...
    float nsd = 2;
    float eta = 2e-6;
    cov.at<float>(0, 0) = pow(xr / 640.0 / nsd, 2.0);
    cov.at<float>(1, 1) = pow(yr / 480.0 / nsd, 2.0);
    cov.at<float>(2, 2) = pow(ar / 180.0 / nsd, 2.0);
    // cov.at<float>(2, 2) = pow(lxr / nsd, 2.0);
    // cov.at<float>(3, 3) = pow(sxr / nsd, 2.0);
    // cov.at<float>(4, 4) = pow(ar / nsd, 2.0);
    // invert
    invert(cov, invCov);
    alpha_ng = alpha_ng * eta;
    cout << "cov: " << invCov.at<float>(0, 0) << " " << invCov.at<float>(1, 1) 
        << " " << invCov.at<float>(2, 2) << invCov.rows << invCov.cols << endl;
    cout << "covariance norm: " << norm(invCov, NORM_L2) << endl;
}

void nac::copyPw(Mat& lrw){
    Mat _pw = lrw.rowRange(0, npY);
    _pw.copyTo(pw);
}


void nac::setUpdateFlag(bool _updateFlag){
   updateFlag = _updateFlag; 
}



