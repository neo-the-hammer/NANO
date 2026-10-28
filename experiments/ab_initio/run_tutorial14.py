#!/usr/bin/env python3
"""Make the ab-initio Hamiltonian of tutorial 14 (fluorinated 12-AGNR).

Tutorial 14 of the ViDES site (vides.nanotcad.com/vides/documentation/
tutorials/tutorial-14-multi-scale-simulations) takes the Hamiltonian of a
nanoribbon from Quantum ESPRESSO and Wannier90.  Its scripts (files.zip:
run_QE, run_Wannier, Wannier2ViDES.f90, environment_variables,
check_failure.sh) are downloaded and run unchanged except for the paths
and the MPI command line; the three files NanoTCAD ViDES needs

    Wannier2ViDES.H   Wannier2ViDES.xyz   n_Nc.dat

are copied to experiments/data/gnr_tutorial14, where the scenario
tut_gnr_fluorine (sims.py, run by R1) reads them.  Needs Linux with apt
(Colab works), about an hour of CPU time, and a network connection.

    python3 experiments/ab_initio/run_tutorial14.py [--work DIR] [--nproc N]

or in a notebook:  %run experiments/ab_initio/run_tutorial14.py
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "gnr_tutorial14")
SITE = "http://vides.nanotcad.com/vides/wp-content/uploads/2012/12/files.zip"
UPF = ("C.pbe-rrkjus.UPF", "F.pbe-rrkjus.UPF")
UPF_URLS = ("https://pseudopotentials.quantum-espresso.org/upf_files/",
            "http://www.quantum-espresso.org/wp-content/uploads/upf_files/",
            "https://web.archive.org/web/2016/http://www.quantum-espresso.org/wp-content/upload/upf_files/")
OUTPUTS = ("Wannier2ViDES.H", "Wannier2ViDES.xyz", "n_Nc.dat")


def sh(cmd, cwd=None, log=None, check=True):
    """Run a shell command, send its output to a log file and show the tail."""
    print("$", cmd, flush=True)
    with open(log or os.devnull, "w") as f:
        rc = subprocess.run(["bash", "-lc", cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if log:
        tail = open(log).read().splitlines()[-12:]
        print("\n".join("   " + l for l in tail), flush=True)
    if check and rc:
        sys.exit("command failed (rc=%d), see %s" % (rc, log))
    return rc


def sub_file(path, pairs):
    txt = open(path).read()
    for pat, rep in pairs:
        txt = re.sub(pat, rep, txt, flags=re.M)
    open(path, "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", default="/content/tut14" if os.path.isdir("/content") else "/tmp/tut14")
    ap.add_argument("--nproc", type=int, default=max(1, os.cpu_count() or 1))
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    pseudo, bindir, scripts = (os.path.join(work, d) for d in ("pseudo", "bin", "files"))

    os.makedirs(work, exist_ok=True)

    # 1. Quantum ESPRESSO, Wannier90, a Fortran compiler and MPI
    sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
       "quantum-espresso wannier90 gfortran openmpi-bin unzip wget", log=os.path.join(work, "apt.log"))
    for exe in ("pw.x", "pw2wannier90.x", "wannier90.x"):
        if not shutil.which(exe):
            sys.exit(exe + " is not installed")

    # 2. The tutorial's scripts
    sh("wget -q -O files.zip %s && unzip -q -o files.zip && rm -rf __MACOSX" % SITE, cwd=work)

    # 3. The pseudopotentials the scripts name
    os.makedirs(pseudo, exist_ok=True)
    os.makedirs(bindir, exist_ok=True)
    for upf in UPF:
        dst = os.path.join(pseudo, upf)
        for base in UPF_URLS:
            sh("wget -q -O %s '%s%s'" % (dst, base, upf), check=False)
            if os.path.exists(dst) and os.path.getsize(dst) > 1000 and b"UPF" in open(dst, "rb").read(2000):
                break
        else:
            sys.exit("could not download %s: put it into %s and run again" % (upf, pseudo))

    # 4. The executables BIN_DIR must hold; the ViDES converter is compiled here
    for exe in ("pw.x", "pw2wannier90.x", "wannier90.x"):
        dst = os.path.join(bindir, exe)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(shutil.which(exe), dst)
    sh("gfortran -o %s/Wannier2ViDES Wannier2ViDES.f90" % bindir, cwd=scripts,
       log=os.path.join(work, "w2v_build.log"))

    # 5. Paths in environment_variables; MPI as root with every core
    sub_file(os.path.join(scripts, "environment_variables"),
             [(r"^PREFIX=.*$", "PREFIX=%s" % work), (r"^BIN_DIR=.*$", "BIN_DIR=%s" % bindir),
              (r"^PSEUDO_DIR=.*$", "PSEUDO_DIR=%s" % pseudo), (r"^TMP_DIR=.*$", "TMP_DIR=%s/tmp" % work)])
    os.makedirs(os.path.join(work, "tmp"), exist_ok=True)
    for name in ("run_QE", "run_Wannier"):
        p = os.path.join(scripts, name)
        sub_file(p, [(r"mpirun -np 8", "mpirun --allow-run-as-root --oversubscribe -np %d" % args.nproc)])
        os.chmod(p, 0o755)
    os.chmod(os.path.join(scripts, "check_failure.sh"), 0o755)

    # 6-7. Ground state and Bloch states, then Wannier90 and the converter
    sh("./run_QE", cwd=scripts, log=os.path.join(work, "run_QE.log"))
    sh("./run_Wannier", cwd=scripts, log=os.path.join(work, "run_Wannier.log"))

    # 8. The three files NanoTCAD ViDES reads
    res = os.path.join(scripts, "results_12AGNR+F")
    os.makedirs(DATA, exist_ok=True)
    for f in OUTPUTS:
        shutil.copy(os.path.join(res, f), DATA)
    print("n, Nc =", open(os.path.join(DATA, "n_Nc.dat")).read().split(),
          "(the tutorial script assumes 28 and 32)")
    print("copied", ", ".join(OUTPUTS), "to", DATA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
