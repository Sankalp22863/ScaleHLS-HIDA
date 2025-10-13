// GPModel.h (header-only, Eigen needed)
// Matern-5/2 + ARD, with fit/update/predict. Works well for ≤ a few hundred points.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>

struct GPModel {
  // hyperparameters
  Eigen::VectorXd lengthscale;        // d (per-dim)
  double sigma_f = 1.0;               // signal std
  double sigma_n = 1e-4;              // noise std

  // training data (normalized)
  Eigen::MatrixXd Xn;                 // n x d  in [0,1]
  Eigen::VectorXd y;                  // n      standardized

  // normalization stats
  Eigen::VectorXd x_min, x_max;       // d
  double y_mean = 0.0, y_std = 1.0;

  // cached factorization K = L L^T
  Eigen::LLT<Eigen::MatrixXd> K_llt;

  // Utils
  static inline double matern52(const Eigen::RowVectorXd& a,
                                const Eigen::RowVectorXd& b,
                                const Eigen::VectorXd& ls,
                                double sf2) {
    Eigen::RowVectorXd z = (a - b).cwiseQuotient(ls.transpose());
    double r2 = std::max(1e-18, z.squaredNorm());
    double r  = std::sqrt(r2);
    double t  = std::sqrt(5.0) * r;
    return sf2 * (1.0 + t + 5.0*r2/3.0) * std::exp(-t);
  }
  static Eigen::MatrixXd toM(const std::vector<std::vector<double>>& X){
    int n=X.size(), d=X[0].size(); Eigen::MatrixXd M(n,d);
    for(int i=0;i<n;++i) for(int j=0;j<d;++j) M(i,j)=X[i][j]; return M;
  }
  static Eigen::VectorXd toV(const std::vector<double>& v){
    Eigen::VectorXd r(v.size()); for(int i=0;i<(int)v.size();++i) r(i)=v[i]; return r;
  }
  Eigen::MatrixXd normalizeX(const Eigen::MatrixXd& X) const {
    Eigen::MatrixXd Z=X;
    for (int j=0;j<X.cols();++j) {
      double den = std::max(1e-12, (x_max(j)-x_min(j)));
      Z.col(j) = (X.col(j).array()-x_min(j)) / den;
    }
    return Z;
  }

  // Fit model to data (X,y) (raw scale).
  void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y_in) {
    const int n=X.size(), d=X[0].size();
    Eigen::MatrixXd Xraw = toM(X);
    Eigen::VectorXd yraw = toV(y_in);

    x_min = Xraw.colwise().minCoeff();
    x_max = Xraw.colwise().maxCoeff();
    Xn    = normalizeX(Xraw);

    y_mean = yraw.mean();
    y_std  = std::max(1e-12, std::sqrt((yraw.array()-y_mean).square().mean()));
    y      = (yraw.array()-y_mean)/y_std;

    if (lengthscale.size()!=d) lengthscale = Eigen::VectorXd::Constant(d, 0.2);

    Eigen::MatrixXd K(n,n);
    const double sf2 = sigma_f*sigma_f;
    for (int i=0;i<n;++i) for (int j=i;j<n;++j) {
      double k = matern52(Xn.row(i), Xn.row(j), lengthscale, sf2);
      K(i,j)=K(j,i)=k;
    }
    K.diagonal().array() += sigma_n*sigma_n;
    K_llt.compute(K);
  }

  // Simple append + refit (fast enough for BO where n is small)
  void update(const std::vector<std::vector<double>>& Xnew,
              const std::vector<double>& ynew) {
    // denormalize old Xn back to raw scale
    Eigen::MatrixXd Xold(Xn.rows(), Xn.cols());
    for (int i=0;i<Xn.rows();++i) for (int j=0;j<Xn.cols();++j)
      Xold(i,j) = Xn(i,j)*(x_max(j)-x_min(j)) + x_min(j);

    // concat
    const int n_old=Xold.rows(), n_new=Xnew.size(), d=Xold.cols();
    Eigen::MatrixXd Xall(n_old+n_new,d); Xall.topRows(n_old)=Xold; Xall.bottomRows(n_new)=toM(Xnew);
    Eigen::VectorXd yall(n_old+n_new);
    for (int i=0;i<n_old;++i) yall(i)= y(i)*y_std + y_mean;
    yall.tail(n_new)= toV(ynew);

    // re-fit
    std::vector<std::vector<double>> Xvec(n_old+n_new, std::vector<double>(d));
    std::vector<double> yvec(n_old+n_new);
    for (int i=0;i<Xall.rows();++i){ for(int j=0;j<d;++j) Xvec[i][j]=Xall(i,j); yvec[i]=yall(i); }
    fit(Xvec, yvec);
  }

  void predict(const std::vector<std::vector<double>>& Xquery,
               std::vector<double>& mu, std::vector<double>& var) const {
    const int nq = Xquery.size();
    Eigen::MatrixXd Xq = toM(Xquery);
    Xq = normalizeX(Xq);

    // k_*  (n_train x nq)
    Eigen::MatrixXd Kxs(Xn.rows(), nq);
    const double sf2 = sigma_f*sigma_f;
    for (int i=0;i<Xn.rows();++i) for (int j=0;j<nq;++j)
      Kxs(i,j) = matern52(Xn.row(i), Xq.row(j), lengthscale, sf2);

    // alpha = K^{-1} y
    Eigen::VectorXd alpha = K_llt.solve(y);
    Eigen::VectorXd mu_q  = Kxs.transpose() * alpha;

    // var = kxx - ||L^{-1}k_*||^2
    Eigen::MatrixXd v = K_llt.matrixL().solve(Kxs);
    Eigen::VectorXd var_q(nq);
    for (int j=0;j<nq;++j) {
      double kxx = sf2;
      var_q(j) = std::max(1e-18, kxx - v.col(j).squaredNorm());
    }

    mu.resize(nq); var.resize(nq);
    for (int i=0;i<nq;++i) { mu[i]=mu_q(i)*y_std + y_mean; var[i]=var_q(i)*y_std*y_std; }
  }
};