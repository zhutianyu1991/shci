#include "../base_system.h"
#include "hamiltonian.h"

#include <cstdio>
#include <unordered_map>
#include <vector>
#include "../util.h"
#include <cmath>

template <class S>
class Green {
 public:
  Green(S& system, Hamiltonian<S>& hamiltonian) : system(system), hamiltonian(hamiltonian) {}

  void run();

 private:
  size_t n_dets;

  size_t n_pdets;

  unsigned n_orbs;

  unsigned n_ao;

  double w;

  std::vector<double> wlist;

  double n;

  bool advanced;

  std::vector<Det> dets_store;

  std::vector<double> coefs_store;

  std::unordered_map<Det, size_t, DetHasher> pdet_to_id;

  S& system;

  Hamiltonian<S>& hamiltonian;

  std::vector<std::vector<std::complex<double>>> G;

  std::vector<std::vector<double>> mo_coeff;

  std::vector<int> orb_idx;

  std::vector<std::vector<double>> bvec;

  std::vector<std::vector<double>> bvec_ao;

  void construct_pdets();

  std::vector<double> construct_b(const unsigned orb);

  std::vector<std::complex<double>> mul_green(const std::vector<std::complex<double>>& vec) const;

  std::vector<std::complex<double>> cg(
      const std::vector<double>& b,
      const std::vector<std::complex<double>>& x0,
	  const std::vector<std::complex<double>>& M,
      const double tol = 1.0e-4);

  std::vector<std::complex<double>> gmres(
      const std::vector<double>& b,
      const std::vector<std::complex<double>>& x0,
	  const std::vector<std::complex<double>>& M,
	  const int mres = 300,
      const double tol = 1.0e-4);

  std::vector<std::complex<double>> bicgstab(
      const std::vector<double>& b,
      const std::vector<std::complex<double>>& x0,
	  const std::vector<std::complex<double>>& M,
      const double tol = 1.0e-4);

  std::complex<double> conjdot(
      const std::vector<std::complex<double>>& va,
	  const std::vector<std::complex<double>>& vb);

  void ApplyRot(
      std::complex<double> &dx, std::complex<double> &dy, 
	  std::complex<double> &cs, std::complex<double> &sn);

  void GenRot(
      std::complex<double> &dx, std::complex<double> &dy, 
	  std::complex<double> &cs, std::complex<double> &sn);

  void Update(
      std::vector<std::complex<double>> &x, int k,
      std::vector<std::vector<std::complex<double>>> h,	
	  std::vector<std::complex<double>> s, std::vector<std::vector<std::complex<double>>> v);

  void output_green();
};

template <class S>
void Green<S>::run() {
  // Store dets and coefs.
  dets_store = system.dets;
  coefs_store = system.coefs;
  n_dets = dets_store.size();
  n_orbs = system.n_orbs;

  // w = Config::get<double>("w_green");
  wlist = Config::get<std::vector<double>>("w_green");
  n = Config::get<double>("n_green");

  // Construct new dets.
  system.dets.clear();
  system.coefs.clear();
  advanced = Config::get<bool>("advanced_green", false);
  if (Parallel::is_master()) {
    if (advanced) {
      printf("Calculating G-\n");
    } else {
      printf("Calculating G+\n");
    }
  }
  construct_pdets();

  // Construct hamiltonian.
  hamiltonian.clear();
  if (advanced) {
    hamiltonian.n_up = hamiltonian.n_up - 1;
  } else {
    hamiltonian.n_up = hamiltonian.n_up + 1;
  }
  hamiltonian.update(system);
  std::cout << "n_up: " << hamiltonian.n_up << std::endl;

  // Get n_ao
  std::ifstream nao_file("n_ao.txt");
  if (nao_file) {
	nao_file >> n_ao;
  }
  else {
	n_ao = n_orbs;
  }
  nao_file.close();
  std::cout << "N_ao: " << n_ao << std::endl;
 
  // Initialize G.
  G.resize(n_ao);
  for (unsigned i = 0; i < n_ao; i++) {
    G[i].assign(n_ao, 0.0);
  }

  std::cout << "n_pdets: " << n_pdets << std::endl;
  bvec.resize(n_orbs);
  for (unsigned i = 0; i < n_orbs; i++) {
    bvec[i].assign(n_pdets, 0.0);
	const auto& bi = construct_b(i);
	for (size_t k = 0; k < n_pdets; k++) {
	  bvec[i][k] = bi[k];
	}
  }

  // Transfer bvec into AO basis
  mo_coeff.resize(n_ao, std::vector<double>(n_orbs, 0.0));
  std::ifstream mo_file("mo_coeff.txt");
  if (mo_file) {
	std::cout << "Transform MO to AO basis" << std::endl;
    double mo_value;
	for (unsigned i = 0; i < n_ao; i++) {
	  for (unsigned j = 0; j < n_orbs; j++) {
	    mo_file >> mo_value;
	    mo_coeff[i][j] = mo_value;
	  }
	}
  }
  else {
	std::cout << "Stay at MO basis" << std::endl;
	for (unsigned i = 0; i < n_orbs; i++) {
	  mo_coeff[i][i] = 1.0;
	}
  }
  mo_file.close();

  bvec_ao.resize(n_ao);
  for (unsigned k = 0; k < n_ao; k++) {
    bvec_ao[k].assign(n_pdets, 0.0);
#pragma omp parallel for
	for (size_t p = 0; p < n_pdets; p++) {
	  for (unsigned i = 0; i < n_orbs; i++) {
        bvec_ao[k][p] += mo_coeff[k][i] * bvec[i][p];
	  }
	}
  }

  // Select orbitals to solve for G
  std::ifstream idx_file("orb_idx.txt");
  if (idx_file) {
	int idx_value;
	while (idx_file >> idx_value) {
	  orb_idx.push_back(idx_value);
	}
  }
  else {
	for (unsigned i = 0; i < n_orbs; i++) {
	  orb_idx.push_back(i);
	}
  }
  idx_file.close();
  std::cout << "G orbs: " << orb_idx.size() << std::endl;

  for (unsigned iw = 0; iw < wlist.size(); iw++) {
    w = wlist[iw];

	std::vector<std::complex<double>> diag(n_pdets, 0.);
    for (size_t k = 0; k < n_pdets; k++) {
      diag[k] = hamiltonian.matrix.get_diag(k);
      if (advanced) {
        diag[k] = w + n * Util::I + (diag[k] - system.energy_var);
	    //diag = w + n * Util::I - (diag - system.energy_var);
      } else {
	    diag[k] = w + n * Util::I - (diag[k] - system.energy_var);
        //diag = w + n * Util::I + (diag - system.energy_var);
      }
    }

	for (unsigned j_idx = 0; j_idx < orb_idx.size(); j_idx++) {
	  int j = orb_idx[j_idx];
	  Timer::checkpoint(Util::str_printf("orb #%zu/%zu @ freq #%zu", j + 1, n_ao, iw+1));
      // Construct bj
      //const auto& bj = construct_b(j);
	  const auto& bj = bvec_ao[j];

      // Generate initial x0.
      std::vector<std::complex<double>> x0(n_pdets, 1.0e-6);

      for (size_t k = 0; k < n_pdets; k++) {
        if (std::abs(bj[k]) > 1.0e-6) {
          //std::complex<double> diag = hamiltonian.matrix.get_diag(k);
          //if (advanced) {
            //diag = w + n * Util::I + (diag - system.energy_var);
	        //diag = w + n * Util::I - (diag - system.energy_var);
          //} else {
		    //diag = w + n * Util::I - (diag - system.energy_var);
            //diag = w + n * Util::I + (diag - system.energy_var);
          //}
          x0[k] = bj[k] / diag[k];
        }
      }

      // Iteratively get H^{-1}bj
      const auto& x = cg(bj, x0, diag);
	  //const auto& x = bicgstab(bj, x0, diag);
	  //const auto& x = gmres(bj, x0, diag);

      for (unsigned i_idx = 0; i_idx < orb_idx.size(); i_idx++) {
        // Dot with bi
        //const auto& bi = construct_b(i);
		int i = orb_idx[i_idx];
        const auto& bi = bvec_ao[i];
	    G[i][j] = Util::dot_omp(bi, x);
      }
    }
    output_green();
  }
}

template <class S>
void Green<S>::construct_pdets() {
  for (size_t i = 0; i < n_dets; i++) {
    Det det = dets_store[i];
    for (unsigned k = 0; k < n_orbs; k++) {
      if (advanced) {  // G-.
        if (det.up.has(k)) {
          det.up.unset(k);
          if (pdet_to_id.count(det) == 0) {
            pdet_to_id[det] = system.dets.size();
            system.dets.push_back(det);
          }
          det.up.set(k);
        }
        //if (det.dn.has(k)) {
          //det.dn.unset(k);
          //if (pdet_to_id.count(det) == 0) {
            //pdet_to_id[det] = system.dets.size();
            //system.dets.push_back(det);
          //}
          //det.dn.set(k);
        //}
      } else {  // G+.
        if (!det.up.has(k)) {
          det.up.set(k);
          if (pdet_to_id.count(det) == 0) {
            pdet_to_id[det] = system.dets.size();
            system.dets.push_back(det);
          }
          det.up.unset(k);
        }
        //if (!det.dn.has(k)) {
          //det.dn.set(k);
          //if (pdet_to_id.count(det) == 0) {
            //pdet_to_id[det] = system.dets.size();
            //system.dets.push_back(det);
          //}
          //det.dn.unset(k);
        //}
      }  // Advanced.
    }
  }
  n_pdets = system.dets.size();
  system.coefs.assign(n_pdets, 0.0);
}

template <class S>
std::vector<double> Green<S>::construct_b(const unsigned j) {
  std::vector<double> b(n_pdets, 0.0);
  int perm_fac;
  for (size_t det_id = 0; det_id < n_dets; det_id++) {
    Det det = dets_store[det_id];
    if (advanced) {  // G-.
      if (j < n_orbs && det.up.has(j)) {
        det.up.unset(j);
        perm_fac = det.up.bit_till(j);
      } else if (j >= n_orbs && det.dn.has(j - n_orbs)) {
        det.dn.unset(j - n_orbs);
        perm_fac = det.dn.bit_till(j - n_orbs);
      } else {
        continue;
      }
    } else {  // G+.
      if (j < n_orbs && !det.up.has(j)) {
        det.up.set(j);
        perm_fac = det.up.bit_till(j);
      } else if (j >= n_orbs && !det.dn.has(j - n_orbs)) {
        det.dn.set(j - n_orbs);
        perm_fac = det.dn.bit_till(j - n_orbs);
      } else {
        continue;
      }
    }  // Advanced.
    const size_t pdet_id = pdet_to_id[det];
    const double coef = coefs_store[det_id];
    perm_fac = (perm_fac % 2 == 0) ? 1 : -1;
    b[pdet_id] = coef * perm_fac;
  }
  return b;
}

template <class S>
void Green<S>::output_green() {
  const auto& filename = Util::str_printf("green_%#.2e_%#.2ei.csv", w, n);

  FILE* file = fopen(filename.c_str(), "w");

  fprintf(file, "i,j,G\n");
  for (unsigned i = 0; i < n_ao; i++) {
    for (unsigned j = 0; j < n_ao; j++) {
      fprintf(file, "%u,%u,%+.10f%+.10fj\n", i, j, G[i][j].real(), G[i][j].imag());
    }
  }

  fclose(file);

  printf("Green's function saved to: %s\n", filename.c_str());
}

template <class S>
std::vector<std::complex<double>> Green<S>::gmres(
    const std::vector<double>& b, const std::vector<std::complex<double>>& x0, 
	const std::vector<std::complex<double>>& M, const int mres, const double tol) {
  std::vector<std::complex<double>> s(mres+1, 0.0);
  std::vector<std::complex<double>> cs(mres+1, 0.0);
  std::vector<std::complex<double>> sn(mres+1, 0.0);
  std::vector<std::complex<double>> r(n_pdets, 0.0);
  std::vector<std::complex<double>> Minvb(n_pdets, 0.0);
  std::vector<std::vector<std::complex<double>>> H(mres+1, std::vector<std::complex<double>>(mres,0.0));
  std::vector<std::vector<std::complex<double>>> Q;
  
  const auto& Ax0 = mul_green(x0);
  std::vector<std::complex<double>> x = x0;

#pragma omp parallel for
  for (size_t i = 0; i < n_pdets; i++) {
    r[i] = (b[i] - Ax0[i]) / M[i];
    Minvb[i] = b[i] / M[i];
  }
  const double beta = std::sqrt(std::abs(Util::dot_omp(r, r)));
  double normb = std::sqrt(std::abs(Util::dot_omp(Minvb, Minvb)));
  if (normb < 1e-10) {
	normb = 1.0;
  }

  double resid = beta;
  if ((resid < tol) || (resid/normb < tol)) {
    printf("Initial GMRES converged: r = %g, r/b = %g\n", resid, resid/normb);
	return x0;
  }

  int maxiter = 400;
  int j = 1;
  while (j <= maxiter) {
	Q.resize(1, std::vector<std::complex<double>>(n_pdets));
	std::fill(s.begin(), s.end(), 0.0);
#pragma omp parallel for
    for (size_t k = 0; k < n_pdets; k++) {
      Q[0][k] = r[k] / beta;
    }
    s[0] = beta;
	
	for (int i = 0; i < mres && j <= maxiter; i++, j++) {
	  std::vector<std::complex<double>> AQ = mul_green(Q[i]);
#pragma omp parallel for
      for (size_t k = 0; k < n_pdets; k++) {
        AQ[k] = AQ[k] / M[k];
      }

	  for (int k = 0; k <= i; k++) {
	    H[k][i] = conjdot(Q[k], AQ);
#pragma omp parallel for
		for (size_t l = 0; l < n_pdets; l++) {
	      AQ[l] -= H[k][i] * Q[k][l];
		}
	  }
	  H[i+1][i] = std::sqrt(std::abs(Util::dot_omp(AQ, AQ)));
	  Q.resize(i+2, std::vector<std::complex<double>>(n_pdets));
#pragma omp parallel for
	  for (size_t l = 0; l < n_pdets; l++) {
		Q[i+1][l] = AQ[l] / H[i+1][i];
	  }

	  for (int k = 0; k < i; k++) {
	    ApplyRot(H[k][i], H[k+1][i], cs[k], sn[k]);
	  }
	  GenRot(H[i][i], H[i+1][i], cs[i], sn[i]);
	  ApplyRot(H[i][i], H[i+1][i], cs[i], sn[i]);
	  ApplyRot(s[i], s[i+1], cs[i], sn[i]);
	  
	  resid = std::abs(s[i+1]);
	  if (j % 20 == 0) printf("Iteration %d: r = %g, r/b = %g\n", j, resid, resid/normb);
	  if ((resid < tol) || (resid/normb < tol)) {
        printf("GMRES converged iter = %d: r = %g, r/b = %g\n", j, resid, resid/normb);
	    Update(x, i, H, s, Q);
		return x;
      }
	}

	Update(x, mres-1, H, s, Q);
    const auto& Ax = mul_green(x);

#pragma omp parallel for
    for (size_t i = 0; i < n_pdets; i++) {
      r[i] = (b[i] - Ax[i]) / M[i];
    }

    const double beta = std::sqrt(std::abs(Util::dot_omp(r, r)));

    double resid = beta;
    if ((resid < tol) || (resid/normb < tol)) {
      printf("GMRES converged iter = %d: r = %g, r/b = %g\n", j, resid, resid/normb);
	  return x;
    }
  }
  printf("GMRES NOT converged iter = %d: r = %g, r/b = %g\n", j, resid, resid/normb);
  return x;
}

template <class S>
void Green<S>::Update(
    std::vector<std::complex<double>> &x, int k,
    std::vector<std::vector<std::complex<double>>> h,	
	std::vector<std::complex<double>> s, std::vector<std::vector<std::complex<double>>> v) {

  std::vector<std::complex<double>> y = s;
  for (int i = k; i >= 0; i--) {
	y[i] = y[i] / h[i][i];
	for (int j = i - 1; j >= 0; j--) {
	  y[j] -= h[j][i] * y[i];
	}
  }

  for (int j = 0; j <= k; j++) {
#pragma omp parallel for
    for (size_t l = 0; l < x.size(); l++) {
	  x[l] += v[j][l] * y[j];
	}
  }
}

template <class S>
void Green<S>::GenRot(
    std::complex<double> &dx, std::complex<double> &dy, 
	std::complex<double> &cs, std::complex<double> &sn) {

  if (std::abs(dy) < 1.0e-10) {
    cs = 1.0;
	sn = 0.0;
  } else {
    std::complex<double> dxysum = std::sqrt(std::conj(dx) * dx + std::conj(dy) * dy);
	cs = dx / dxysum;
	sn = dy / dxysum;
  }
}

template <class S>
void Green<S>::ApplyRot(
    std::complex<double> &dx, std::complex<double> &dy, 
	std::complex<double> &cs, std::complex<double> &sn) {

  std::complex<double> tmp = std::conj(cs) * dx + std::conj(sn) * dy;
  dy = -sn * dx + cs * dy;
  dx = tmp;
}

template <class S>
std::complex<double> Green<S>::conjdot(
    const std::vector<std::complex<double>>& va, 
	const std::vector<std::complex<double>>& vb) {

  std::vector<std::complex<double>> vaconj(va.size(), 0.0);
#pragma omp parallel for
  for (size_t i = 0; i < va.size(); i++) {
    vaconj[i] = std::conj(va[i]);
  }

  const std::complex<double>& vab = Util::dot_omp(vaconj, vb);
  return vab;
}

template <class S>
std::vector<std::complex<double>> Green<S>::bicgstab(
    const std::vector<double>& b, const std::vector<std::complex<double>>& x0, 
	const std::vector<std::complex<double>>& M, const double tol) {
  std::vector<std::complex<double>> x(n_pdets, 0.0);
  std::vector<std::complex<double>> r(n_pdets, 0.0);
  std::vector<std::complex<double>> p(n_pdets, 0.0);
  std::vector<std::complex<double>> y(n_pdets, 0.0);
  std::vector<std::complex<double>> v(n_pdets, 0.0);
  std::vector<std::complex<double>> rhat(n_pdets, 0.0);
  std::vector<std::complex<double>> svec(n_pdets, 0.0);
  std::vector<std::complex<double>> z(n_pdets, 0.0);

  const auto& Ax0 = mul_green(x0);

#pragma omp parallel for
  for (size_t i = 0; i < n_pdets; i++) {
    r[i] = b[i] - Ax0[i];
    rhat[i] = r[i];
    x[i] = x0[i];
  }

  double normb = std::sqrt(std::abs(Util::dot_omp(b, b)));
  double residual = std::sqrt(std::abs(Util::dot_omp(r, r)));
  if ((residual < tol) || (residual/normb < tol)) {
    printf("Initial vector converged: r = %g, r/b = %g\n", residual, residual/normb);
	return x;
  }

  const std::complex<double>& rho = 1.0;
  const std::complex<double>& alpha = 1.0;
  const std::complex<double>& omega = 1.0;
  int iter = 0;
  while ((residual > tol) && (residual/normb > tol)) {
    const std::complex<double>& rhonew = conjdot(rhat, r); 
    if (std::abs(rhonew) < 1.0e-10) throw std::runtime_error("rho goes to 0");  
	const std::complex<double>& beta = (rhonew / rho) * (alpha / omega); 
    const std::complex<double>& rho = rhonew;
#pragma omp parallel for
	for (size_t i = 0; i < n_pdets; i++) {
	  p[i] = r[i] + beta * (p[i] - omega * v[i]);
	  y[i] = p[i] / M[i];
	}

	v = mul_green(y);
	const std::complex<double>& alpha = rho / conjdot(rhat, v);
#pragma omp parallel for
	for (size_t i = 0; i < n_pdets; i++) {
	  svec[i] = r[i] - alpha * v[i];
	  x[i] = x[i] + alpha * y[i];
	}

	residual = std::sqrt(std::abs(Util::dot_omp(svec, svec)));
	if ((residual < tol) || (residual/normb < tol)) {
	  break;
	}

#pragma omp parallel for
	for (size_t i = 0; i < n_pdets; i++) {
	  z[i] = svec[i] / M[i];
	}
	const auto& Az = mul_green(z);
	const std::complex<double>& omega = conjdot(Az, svec) / conjdot(Az, Az);

#pragma omp parallel for
	for (size_t i = 0; i < n_pdets; i++) {
	  x[i] = x[i] + omega * z[i];
	  r[i] = svec[i] - omega * Az[i];
	}

	residual = std::sqrt(std::abs(Util::dot_omp(r, r)));
	iter++;
    if (iter % 20 == 0) printf("Iteration %d: r = %g, r/b = %g\n", iter, residual, residual/normb);
    if (iter > 400) throw std::runtime_error("bicgstab does not converge");
  }

  printf("Final iteration %d: r = %g, r/b = %g\n", iter, residual, residual/normb);

  return x;
}

template <class S>
std::vector<std::complex<double>> Green<S>::cg(
    const std::vector<double>& b, const std::vector<std::complex<double>>& x0, 
	const std::vector<std::complex<double>>& M, const double tol) {
  std::vector<std::complex<double>> x(n_pdets, 0.0);
  std::vector<std::complex<double>> r(n_pdets, 0.0);
  std::vector<std::complex<double>> p(n_pdets, 0.0);
  std::vector<std::complex<double>> z(n_pdets, 0.0);
  std::vector<std::complex<double>> xsave(n_pdets, 0.0);

  const auto& Ax0 = mul_green(x0);

#pragma omp parallel for
  for (size_t i = 0; i < n_pdets; i++) {
    r[i] = b[i] - Ax0[i];
    z[i] = r[i] / M[i];
  }
  p = z;
  x = x0;

  double resid = std::sqrt(std::abs(Util::dot_omp(r, r)));
  double normb = std::sqrt(std::abs(Util::dot_omp(b, b)));
  if (normb < 1e-10) {
	normb = 1.0;
  }
  double residsave = resid;

  if ((resid < tol) || (resid/normb < tol)) {
    printf("Initial CG converged: r = %g, r/b = %g\n", resid, resid/normb);
	return x0;
  }

  int iter = 0;
  int maxiter = 500;
  while (resid > tol && iter < maxiter) {
    //const std::complex<double>& rTr = Util::dot_omp(r, r);
	const std::complex<double>& rTz = Util::dot_omp(r, z);
    const auto& Ap = mul_green(p);
    const std::complex<double>& pTAp = Util::dot_omp(p, Ap);
    const std::complex<double>& a = rTz / pTAp;
#pragma omp parallel for
    for (size_t j = 0; j < n_pdets; j++) {
      x[j] += a * p[j];
      r[j] -= a * Ap[j];
	  z[j] = r[j] / M[j];
    }
    const std::complex<double>& rTr = Util::dot_omp(r, r);
    const std::complex<double>& rTz_new = Util::dot_omp(r, z);
	const std::complex<double>& beta = rTz_new / rTz;
#pragma omp parallel for
    for (size_t j = 0; j < n_pdets; j++) {
      p[j] = z[j] + beta * p[j];
    }

    resid = std::sqrt(std::abs(rTr));
	if (resid < residsave) {
	  residsave = resid;
#pragma omp parallel for
      for (size_t j = 0; j < n_pdets; j++) {
	    xsave[j] = x[j];
	  }
	}

	iter++;
    if (iter % 20 == 0) printf("Iteration %d: r = %g, r/b = %g\n", iter, resid, resid/normb);
    if (iter == maxiter) printf("cg does not converge ");
	if (iter == maxiter && residsave > 0.1) printf("Resid Too Large!!!\n");
  }

  printf("Smallest resid: r = %g, r/b = %g\n", residsave, residsave/normb);

  if (iter < maxiter){
    return x;
  }
  else {
	return xsave;
  }
}

template <class S>
std::vector<std::complex<double>> Green<S>::mul_green(
    const std::vector<std::complex<double>>& vec) const {
  auto G_vec = hamiltonian.matrix.mul(vec);

  for (size_t i = 0; i < n_pdets; i++) {
    if (advanced) {
        // G_vec[i] = (w + n * Util::I + system.energy_var) * vec[i] - G_vec[i];
		G_vec[i] = (w + n * Util::I - system.energy_var) * vec[i] + G_vec[i];
    } else {
        // G_vec[i] = (w + n * Util::I - system.energy_var) * vec[i] + G_vec[i];
        G_vec[i] = (w + n * Util::I + system.energy_var) * vec[i] - G_vec[i];
	}
  }
  return G_vec;
}
