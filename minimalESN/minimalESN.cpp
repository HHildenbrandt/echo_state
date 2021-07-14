#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>


Eigen::VectorXd minimalESN(const Eigen::VectorXd& data)
{
  Eigen::MatrixXd WR(10, 10);
  WR <<
    -0.36652994, 0.7930877, -0.1415693, 0.2762202, -0.380918437, 0.06596489, 0.2731735, -0.5512403, -0.28644420, 0.10486813,
    -0.39202286, -0.7999955, 0.1786538, 0.4633583, -0.684657824, -0.03822973, -0.6351252, 0.2132468, 0.66249950, 0.15105157,
    -0.16043866, -0.2351548, -0.8775927, 0.3095554, -0.741399020, -0.26074246, 0.2726691, 0.2839211, 0.60920999, -0.29036704,
    -0.17404446, 0.1951356, -0.5251789, 0.5390910, 0.053129330, -0.69206819, -0.5148891, 0.3879448, 0.11181810, -0.43132334,
    -0.20185923, 0.8393748, -0.1801699, 0.7100308, 0.002869283, -0.79197058, -0.5714905, 0.8403981, 0.30247652, -0.69108494,
    -0.18533557, -0.4711514, 0.6436871, -0.3765919, -0.084958298, 0.73650529, -0.4122342, -0.1172462, -0.88377642, 0.28412026,
    -0.03825873, 0.2313482, -0.4751556, -0.3732474, 0.687641974, -0.03823147, -0.8567587, -0.3573279, 0.52477910, -0.01382085,
    0.45854326, 0.7396408, 0.2609720, 0.5835236, -0.623044931, 0.22920070, 0.7026408, 0.8659291, -0.04714639, -0.26016566,
    -0.34831818, 0.3595969, -0.8889803, -0.2365551, 0.056430113, -0.36427703, 0.3619058, 0.8702820, -0.11944282, -0.27834089,
    0.35606102, -0.4120551, 0.4828909, -0.9128946, 0.277018075, 0.58529811, -0.2625352, -0.8223866, 0.44606517, -0.13953548;

  Eigen::MatrixXd WRin(10, 2);
  WRin <<
    0.37449324, -0.27407792,
    -0.21120992, 0.49315817,
    0.44721172, -0.27514793,
    0.38452071, 0.04659857,
    -0.49132165, 0.33735363,
    0.02189195, 0.32439996,
    -0.10129456, 0.36420730,
    0.46001960, 0.20136010,
    0.31390361, 0.49464463,
    -0.40755102, 0.06215099;

  Eigen::VectorXd WRout(12);
  WRout << 0.666955, 1.184691, -18.1748, 1.524577, 3.411875, 13.14072, 3.0188, 9.743195, -3.428125, -5.859031, 0.3323515, 11.80838;
  
  Eigen::VectorXd xR(10);
  xR <<
    -0.32542414,
    0.09196801,
    0.62923045,
    0.97357675,
    0.98471708,
    -0.94304354,
    -0.03772297,
    0.80870724,
    0.65911595,
    -0.93380501;


  const size_t trainLen = 2000;
  const size_t testLen = 2000;
  const size_t initLen = 100;

  // generate the ESN reservoir
  const size_t inSize = 1;
  const size_t outSize = 1;
  const size_t resSize = 10;
  const double a = 1.0; // leaking rate

  // create matrices filled with random numbers in (-0.5, +0.5)
  // Note: w/o explicit construction Win and W would be an expression templates
  auto Win = Eigen::MatrixXd(0.5 * Eigen::MatrixXd::Random(resSize, inSize + 1));
  auto W = Eigen::MatrixXd(0.5 * Eigen::MatrixXd::Random(resSize, resSize));

  //# normalizing and setting spectral radius
  std::cout << "Computing spectral radius: ";
  auto rhoW = W.eigenvalues().cwiseAbs()(0);    // biggest abs eigenvalue
  W *= 1.25 / rhoW;
  std::cout << rhoW << '\n';

  // allocated memory for the design(collected states) matrix
  auto X = Eigen::MatrixXd::Zero(1 + inSize + resSize, trainLen -initLen).eval();
  
  // run the reservoir with the data and collect X
  auto x = Eigen::MatrixXd(resSize, 1);
  x *= 0.0;
  auto u = Eigen::Vector2d({ 1, 0 });
  for (size_t t = 0; t < trainLen; ++t) {
    u(1) = data(t);
    x = (1.0 -a) * x + (a * (Win * u + W * x).array().tanh()).matrix();
    if (t >= initLen) {
      auto&& col = X.col(t -initLen);    // '&&' is crucial
      col(0) = u(0); col(1) = u(1);
      col.block(2, 0, resSize, 1) = x;
    }
  }

  // train output network
  auto Yt = data.block(initLen + 1, 0, trainLen -initLen, 1);
  auto XX = (X * X.transpose()).eval();
  XX += Eigen::VectorXd::Constant(XX.cols(), 1e-8).asDiagonal();
  auto Wout = ((X * Yt).transpose() * XX.inverse()).eval();

  auto Y = Eigen::VectorXd(testLen);
  u(1) = data[trainLen];
  auto xx = Eigen::MatrixXd(resSize + 2, 1);
  for (int t = 0; t < testLen; ++t) {
    x = (1.0 - a) * x + (a * (Win * u + W * x).array().tanh()).matrix();
    xx(0) = 1; xx(1) = u(1);
    xx.block(2, 0, resSize, 1) = x;
    auto y = (Wout * xx)(0);
    Y(t) = u(1) = y;
  }
  std::cout << Y << std::endl;
  return Y;
}


int main()
{
  try {
    std::random_device rd;
    std::srand(rd());   // Eigen uses srand()!?
    auto is = std::ifstream{ "../../MackeyGlass_t17.txt" };
    if (!is) {
      throw std::runtime_error("Input timeseries not found :(");
    }
    // stream file into a std::vector
    auto ts = std::vector<double>{};
    std::copy(std::istream_iterator<double>(is), std::istream_iterator<double>{}, std::back_inserter(ts));
    // create a view into the std::vector
    auto timeseries = Eigen::Map<Eigen::VectorXd>(ts.data(), ts.size());
    auto res = minimalESN(timeseries);    // we can pass the view because Eigen::Map<T>s are derived from T
    return 0;
  }
  catch (std::exception& err) {
    std::cerr << err.what() << "\nBailing out.\n";
  }
  return -1;
}
