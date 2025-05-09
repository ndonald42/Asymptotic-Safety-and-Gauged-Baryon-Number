import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
fg = 0.1
fy = 0.1
flam = 0.1
pi = np.pi

# Define the system of differential equations below the Planck scale
def system_below(t, y):
    g1, g2, g3, gB, g12, yt, yb, y1, y2, y3, k1, k2, k3, L, Lp, Lm = y
    beta = 1 / (16 * pi**2)
    beta_lambda = 1 / (4 * pi)**2
    
    dg1_dt = beta * (77/10 * g1**3)
    dg2_dt = beta * (-7/6 * g2**3)
    dg3_dt = beta * (-7 * g3**3)
    dgB_dt = beta * (11 * gB**3 + 77/6 * gB * g12**2 - 16/3 * gB**2 * g12)
    dg12_dt = beta * (-16/5 * g1**2 * gB - 16/3 * gB * g12**2 + 77/5 * g1**2 * g12 +
                      11 * gB**2 * g12 + 77/6 * g12**3)
    dyt_dt = beta * (9/2 * yt**3 + 3/2 * yb**2 * yt + 3 * k1**2 * yt + 3 * k2**2 * yt +
                     3 * k3**2 * yt - 17/20 * g1**2 * yt - 9/4 * g2**2 * yt -
                     8 * g3**2 * yt - 2/3 * gB**2 * yt - 5/3 * gB * g12 * yt -
                     17/12 * g12**2 * yt)
    dyb_dt = beta * (3/2 * yt**2 * yb + 9/2 * yb**3 + 3 * k1**2 * yb + 3 * k2**2 * yb +
                     3 * k3**2 * yb - 1/4 * g1**2 * yb - 9/4 * g2**2 * yb -
                     8 * g3**2 * yb - 2/3 * gB**2 * yb + 1/3 * gB * g12 * yb -
                     5/12 * g12**2 * yb)
    dy1_dt = beta * (7 * y1**3 + 1/2 * k1**2 * y1 + 3 * y2**2 * y1 + 3 * y3**2 * y1 -
                     9/10 * g1**2 * y1 - 9/2 * g2**2 * y1 - 3 * gB**2 * y1 +
                     3 * gB * g12 * y1 - 3/2 * g12**2 * y1)
    dy2_dt = beta * (4 * y2**3 + k2**2 * y2 + 6 * y1**2 * y2 + 3 * y3**2 * y2 -
                     18/5 * g1**2 * y2 - 3 * gB**2 * y2 + 6 * gB * g12 * y2 -
                     6 * g12**2 * y2)
    dy3_dt = beta * (4 * y3**3 + k3**2 * y3 + 6 * y1**2 * y3 + 3 * y2**2 * y3 -
                     3 * gB**2 * y3)
    dk1_dt = beta * (3 * yt**2 * k1 + 3 * yb**2 * k1 + 1/2 * y1**2 * k1 + 9/2 * k1**3 +
                     3 * k2**2 * k1 + 3 * k3**2 * k1 - 9/4 * g1**2 * k1 -
                     9/4 * g2**2 * k1 - 15/4 * g12**2 * k1)
    dk2_dt = beta * (3 * yt**2 * k2 + 3 * yb**2 * k2 + 1/2 * y2**2 * k2 + 3 * k1**2 * k2 +
                     9/2 * k2**3 + 3/2 * k3**2 * k2 - 9/4 * g1**2 * k2 -
                     9/4 * g2**2 * k2 - 15/4 * g12**2 * k2)
    dk3_dt = beta * (3 * yt**2 * k3 + 3 * yb**2 * k3 + 1/2 * y3**2 * k3 + 3 * k1**2 * k3 +
                     3/2 * k2**2 * k3 + 9/2 * k3**3 - 9/20 * g1**2 * k3 -
                     9/4 * g2**2 * k3 - 3/4 * g12**2 * k3)
    dL_dt = beta_lambda * (12 * L**2 + 2 * Lm**2 - 9/5 * g1**2 * L - 3 * g12**2 * L -
                           9 * g2**2 * L + 27/100 * g1**4 + 9/10 * g1**2 * g12**2 +
                           9/10 * g1**2 * g2**2 + 3/4 * g12**4 + 3/2 * g2**2 * g12**2 +
                           9/4 * g2**4 + 12 * yt**2 * L + 12 * yb**2 * L + 12 * k1**2 * L +
                           12 * k2**2 * L + 12 * k3**2 * L - 12 * yt**4 - 12 * yb**4 -
                           12 * k1**4 - 12 * k2**4 - 12 * k3**4)
    dLp_dt = beta_lambda * (10 * Lp**2 + 4 * Lm**2 - 12 * gB**2 * Lp + 12 * gB**4 +
                            24 * y1**2 * Lp + 12 * y2**2 * Lp + 12 * y3**2 * Lp -
                            24 * y1**4 - 12 * y2**4 - 12 * y3**4)
    dLm_dt = beta_lambda * (6 * L * Lm + 4 * Lm * Lp + 4 * Lm**2 - 9/10 * g1**2 * Lm -
                            6 * gB**2 * Lm - 3/2 * g12**2 * Lm - 9/2 * g2**2 * Lm +
                            3 * g12**2 * gB**2 + 6 * yt**2 * Lm + 6 * yb**2 * Lm +
                            12 * y1**2 * Lm + 6 * y2**2 * Lm + 6 * y3**2 * Lm +
                            6 * k1**2 * Lm + 6 * k2**2 * Lm + 6 * k3**2 * Lm -
                            12 * y1**2 * k1**2 - 12 * y2**2 * k2**2 - 12 * y3**2 * k3**2)
    
    return [dg1_dt, dg2_dt, dg3_dt, dgB_dt, dg12_dt, dyt_dt, dyb_dt, dy1_dt, dy2_dt,
            dy3_dt, dk1_dt, dk2_dt, dk3_dt, dL_dt, dLp_dt, dLm_dt]

# Define the system of differential equations above the Planck scale
def system_above(t, y):
    g1n, g2n, g3n, gBn, g12n, ytn, ybn, y1n, y2n, y3n, k1n, k2n, k3n, Ln, Lpn, Lmn = y
    beta = 1 / (16 * pi**2)
    beta_lambda = 1 / (4 * pi)**2
    
    dg1n_dt = beta * (77/10 * g1n**3) - fg * g1n
    dg2n_dt = beta * (-7/6 * g2n**3) - fg * g2n
    dg3n_dt = beta * (-7 * g3n**3) - fg * g3n
    dgBn_dt = beta * (11 * gBn**3 + 77/6 * gBn * g12n**2 - 16/3 * gBn**2 * g12n) - fg * gBn
    dg12n_dt = beta * (-16/5 * g1n**2 * gBn - 16/3 * gBn * g12n**2 + 77/5 * g1n**2 * g12n +
                       11 * gBn**2 * g12n + 77/6 * g12n**3) - fg * g12n
    dytn_dt = beta * (9/2 * ytn**3 + 3/2 * ybn**2 * ytn + 3 * k1n**2 * ytn + 3 * k2n**2 * ytn +
                      3 * k3n**2 * ytn - 17/20 * g1n**2 * ytn - 9/4 * g2n**2 * ytn -
                      8 * g3n**2 * ytn - 2/3 * gBn**2 * ytn - 5/3 * gBn * g12n * ytn -
                      17/12 * g12n**2 * ytn) - fy * ytn
    dybn_dt = beta * (3/2 * ytn**2 * ybn + 9/2 * ybn**3 + 3 * k1n**2 * ybn + 3 * k2n**2 * ybn +
                      3 * k3n**2 * ybn - 1/4 * g1n**2 * ybn - 9/4 * g2n**2 * ybn -
                      8 * g3n**2 * ybn - 2/3 * gBn**2 * ybn + 1/3 * gBn * g12n * ybn -
                      5/12 * g12n**2 * ybn) - fy * ybn
    dy1n_dt = beta * (7 * y1n**3 + 1/2 * k1n**2 * y1n + 3 * y2n**2 * y1n + 3 * y3n**2 * y1n -
                      9/10 * g1n**2 * y1n - 9/2 * g2n**2 * y1n - 3 * gBn**2 * y1n +
                      3 * gBn * g12n * y1n - 3/2 * g12n**2 * y1n) - fy * y1n
    dy2n_dt = beta * (4 * y2n**3 + k2n**2 * y2n + 6 * y1n**2 * y2n + 3 * y3n**2 * y2n -
                      18/5 * g1n**2 * y2n - 3 * gBn**2 * y2n + 6 * gBn * g12n * y2n -
                      6 * g12n**2 * y2n) - fy * y2n
    dy3n_dt = beta * (4 * y3n**3 + k3n**2 * y3n + 6 * y1n**2 * y3n + 3 * y2n**2 * y3n -
                      3 * gBn**2 * y3n) - fy * y3n
    dk1n_dt = beta * (3 * ytn**2 * k1n + 3 * ybn**2 * k1n + 1/2 * y1n**2 * k1n + 9/2 * k1n**3 +
                      3 * k2n**2 * k1n + 3 * k3n**2 * k1n - 9/4 * g1n**2 * k1n -
                      9/4 * g2n**2 * k1n - 15/4 * g12n**2 * k1n) - fy * k1n
    dk2n_dt = beta * (3 * ytn**2 * k2n + 3 * ybn**2 * k2n + 1/2 * y2n**2 * k2n + 3 * k1n**2 * k2n +
                      9/2 * k2n**3 + 3/2 * k3n**2 * k2n - 9/4 * g1n**2 * k2n -
                      9/4 * g2n**2 * k2n - 15/4 * g12n**2 * k2n) - fy * k2n
    dk3n_dt = beta * (3 * ytn**2 * k3n + 3 * ybn**2 * k3n + 1/2 * y3n**2 * k3n + 3 * k1n**2 * k3n +
                      3/2 * k2n**2 * k3n + 9/2 * k3n**3 - 9/20 * g1n**2 * k3n -
                      9/4 * g2n**2 * k3n - 3/4 * g12n**2 * k3n) - fy * k3n
    dLn_dt = beta_lambda * (12 * Ln**2 + 2 * Lmn**2 - 9/5 * g1n**2 * Ln - 3 * g12n**2 * Ln -
                            9 * g2n**2 * Ln + 27/100 * g1n**4 + 9/10 * g1n**2 * g12n**2 +
                            9/10 * g1n**2 * g2n**2 + 3/4 * g12n**4 + 3/2 * g2n**2 * g12n**2 +
                            9/4 * g2n**4 + 12 * ytn**2 * Ln + 12 * ybn**2 * Ln + 12 * k1n**2 * Ln +
                            12 * k2n**2 * Ln + 12 * k3n**2 * Ln - 12 * ytn**4 - 12 * ybn**4 -
                            12 * k1n**4 - 12 * k2n**4 - 12 * k3n**4) - flam * Ln
    dLpn_dt = beta_lambda * (10 * Lpn**2 + 4 * Lmn**2 - 12 * gBn**2 * Lpn + 12 * gBn**4 +
                             24 * y1n**2 * Lpn + 12 * y2n**2 * Lpn + 12 * y3n**2 * Lpn -
                             24 * y1n**4 - 12 * y2n**4 - 12 * y3n**4) - flam * Lpn
    dLmn_dt = beta_lambda * (6 * Ln * Lmn + 4 * Lmn * Lpn + 4 * Lmn**2 - 9/10 * g1n**2 * Lmn -
                             6 * gBn**2 * Lmn - 3/2 * g12n**2 * Lmn - 9/2 * g2n**2 * Lmn +
                             3 * g12n**2 * gBn**2 + 6 * ytn**2 * Lmn + 6 * ybn**2 * Lmn +
                             12 * y1n**2 * Lmn + 6 * y2n**2 * Lmn + 6 * y3n**2 * Lmn +
                             6 * k1n**2 * Lmn + 6 * k2n**2 * Lmn + 6 * k3n**2 * Lmn -
                             12 * y1n**2 * k1n**2 - 12 * y2n**2 * k2n**2 - 12 * y3n**2 * k3n**2) - flam * Lmn
    
    return [dg1n_dt, dg2n_dt, dg3n_dt, dgBn_dt, dg12n_dt, dytn_dt, dybn_dt, dy1n_dt, dy2n_dt,
            dy3n_dt, dk1n_dt, dk2n_dt, dk3n_dt, dLn_dt, dLpn_dt, dLmn_dt]

# Initial conditions for below the Planck scale (t = 0)
y0_below = [0.46738, 0.63829, 1.05737, 0.3, 0.14988 - 0.00000037335, 0.85322, 0.01388,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25828, 0.2, -0.004]

# Solve the system below the Planck scale (t = 0 to t = 37.0237)
t_below = (0, 37.0237)
sol_below = solve_ivp(system_below, t_below, y0_below, method='RK45', rtol=1e-12, atol=1e-15)

# Extract solutions at t = 37.0237 for initial conditions above the Planck scale
t_eval = np.linspace(0, 37.0237, 1000)
sol_below_eval = solve_ivp(system_below, t_below, y0_below, method='RK45', t_eval=t_eval,
                           rtol=1e-12, atol=1e-15)
y0_above = sol_below_eval.y[:, -1]  # Values at t = 37.0237

# Solve the system above the Planck scale (t = 37.0237 to t = 137.0237)
t_above = (37.0237, 137.0237)
sol_above = solve_ivp(system_above, t_above, y0_above, method='RK45', rtol=1e-12, atol=1e-15)

# Combine solutions for plotting
t_above_eval = np.linspace(37.0237, 137.0237, 1000)
sol_above_eval = solve_ivp(system_above, t_above, y0_above, method='RK45', t_eval=t_above_eval,
                           rtol=1e-12, atol=1e-15)

# Plotting
# Gauge couplings plot
plt.figure(figsize=(10, 6))
plt.plot(sol_below_eval.t, sol_below_eval.y[0], label=r'$g_1$')
plt.plot(sol_below_eval.t, sol_below_eval.y[1], label=r'$g_2$')
plt.plot(sol_below_eval.t, sol_below_eval.y[2], label=r'$g_3$')
plt.plot(sol_below_eval.t, sol_below_eval.y[3], label=r'$g_B$')
plt.plot(sol_below_eval.t, sol_below_eval.y[4], label=r'$\tilde{g}$')
plt.plot(sol_above_eval.t, sol_above_eval.y[0], label=r'$g_1$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[1], label=r'$g_2$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[2], label=r'$g_3$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[3], label=r'$g_B$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[4], label=r'$\tilde{g}$', linestyle='--')
plt.xlabel(r'$\ln(\mu / 1 \, \mathrm{TeV})$')
plt.ylabel('Gauge Couplings')
plt.grid(True)
plt.legend()
plt.title('Gauge Couplings Evolution')
plt.show()

# Yukawa couplings plot 1 (yt, yb)
plt.figure(figsize=(10, 6))
plt.plot(sol_below_eval.t, sol_below_eval.y[5], label=r'$y_t$')
plt.plot(sol_below_eval.t, sol_below_eval.y[6], label=r'$y_b$')
plt.plot(sol_above_eval.t, sol_above_eval.y[5], label=r'$y_t$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[6], label=r'$y_b$', linestyle='--')
plt.xlabel(r'$\ln(\mu / 1 \, \mathrm{TeV})$')
plt.ylabel('Yukawa Couplings')
plt.grid(True)
plt.legend()
plt.title('Yukawa Couplings (yt, yb) Evolution')
plt.show()

# Yukawa couplings plot 2 (y1, y2, y3)
plt.figure(figsize=(10, 6))
plt.plot(sol_below_eval.t, sol_below_eval.y[7], label=r'$y_1$')
plt.plot(sol_below_eval.t, sol_below_eval.y[8], label=r'$y_2$')
plt.plot(sol_below_eval.t, sol_below_eval.y[9], label=r'$y_3$')
plt.plot(sol_above_eval.t, sol_above_eval.y[7], label=r'$y_1$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[8], label=r'$y_2$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[9], label=r'$y_3$', linestyle='--')
plt.xlabel(r'$\ln(\mu / 1 \, \mathrm{TeV})$')
plt.ylabel('Yukawa Couplings')
plt.grid(True)
plt.legend()
plt.title('Yukawa Couplings (y1, y2, y3) Evolution')
plt.show()

# Yukawa couplings plot 3 (k1, k2, k3)
plt.figure(figsize=(10, 6))
plt.plot(sol_below_eval.t, sol_below_eval.y[10], label=r'$\kappa_1$')
plt.plot(sol_below_eval.t, sol_below_eval.y[11], label=r'$\kappa_2$')
plt.plot(sol_below_eval.t, sol_below_eval.y[12], label=r'$\kappa_3$')
plt.plot(sol_above_eval.t, sol_above_eval.y[10], label=r'$\kappa_1$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[11], label=r'$\kappa_2$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[12], label=r'$\kappa_3$', linestyle='--')
plt.xlabel(r'$\ln(\mu / 1 \, \mathrm{TeV})$')
plt.ylabel('Yukawa Couplings')
plt.grid(True)
plt.legend()
plt.title('Yukawa Couplings (k1, k2, k3) Evolution')
plt.show()

# Scalar couplings plot (L, Lp, Lm)
plt.figure(figsize=(10, 6))
plt.plot(sol_below_eval.t, sol_below_eval.y[13], label=r'$\lambda$')
plt.plot(sol_below_eval.t, sol_below_eval.y[14], label=r'$\lambda_p$')
plt.plot(sol_below_eval.t, sol_below_eval.y[15], label=r'$\lambda_m$')
plt.plot(sol_above_eval.t, sol_above_eval.y[13], label=r'$\lambda$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[14], label=r'$\lambda_p$', linestyle='--')
plt.plot(sol_above_eval.t, sol_above_eval.y[15], label=r'$\lambda_m$', linestyle='--')
plt.xlabel(r'$\ln(\mu / 1 \, \mathrm{TeV})$')
plt.ylabel('Scalar Couplings')
plt.grid(True)
plt.legend()
plt.title('Scalar Couplings Evolution')
plt.show()

