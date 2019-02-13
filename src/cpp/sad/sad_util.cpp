#include "sad_util.hpp"
#include "../util/string.hpp"

namespace lightspeed {
    
int SADAtom::nfrzc() const
{
    int val = 0;
    for (size_t ind = 0; ind < Ls_.size(); ind++) {
        if (!acts_[ind]) val += 2 * Ls_[ind] + 1;
    }
    return val;
}
int SADAtom::nact() const
{
    int val = 0;
    for (size_t ind = 0; ind < Ls_.size(); ind++) {
        if (acts_[ind]) val += 2 * Ls_[ind] + 1;
    }
    return val;
}

std::vector<double> SADAtom::nocc(double npair) const
{
    // NOTE: This explicitly assumes that the frz/act orbitals are ordered.
    // This is the new SAD convention but is not a repr. inv. so I'll check it
    // for now. TODO: Fix this to actually return arbirarily-ordered nocc.
    bool found_act = false;
    for (bool act : acts_) {
        if (act && !found_act) found_act = false;
        if (!act && found_act) throw std::runtime_error("SADAtom::nocc: function is invalid.");
    }
    // Yech.

    std::vector<double> val(ntot());
    if (npair == 0.0) {
        // Ghost atom
    } else if (npair < nfrzc()) {
        // Cation wing
        int nmax = nfrzc();
        for (int ind = 0; ind < nmax; ind++) {
            val[ind] = 1.0;
        }
    } else if (npair > ntot()) {
        // Anion wing
        int nmax = ntot();
        for (int ind = 0; ind < nmax; ind++) {
            val[ind] = 1.0;
        }
    } else {
        // Valence sweet spot
        int nmax = nfrzc();
        for (int ind = 0; ind < nmax; ind++) {
            val[ind] = 1.0;
        }
        double nval = (npair - nmax) / (nact());
        int nmax2 = ntot();
        for (int ind = nmax; ind < nmax2; ind++) {
            val[ind] = nval; 
        }
    }
    return val;
}
    
std::vector<SADAtom> SADAtom::atom_list__;
const SADAtom& SADAtom::get(int N)
{
    if (!atom_list__.size()) atom_list__ = build_atom_list();
    if (N >= atom_list__.size()) {
        throw std::runtime_error("SADAtom::get: atom N is too large - maybe you have found the island of stability.");
    }
    return atom_list__[N];
}

std::vector<SADAtom> SADAtom::build_atom_list()
{
    std::vector<SADAtom> atom_list;
    
    // Gh [][]
    atom_list.push_back(SADAtom(0, {}, {}));
    // H  - He: [][1s]
    for (int N = 1; N <= 2; N++) {
        atom_list.push_back(SADAtom(N, {0}, {true}));
    }
    // Li - Be: [1s][2s]
    for (int N = 3; N <= 4; N++) {
        atom_list.push_back(SADAtom(N, {0, 0}, {false, true}));
    }
    // B  - Ne: [1s][2s2p]
    for (int N = 5; N <= 10; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1}, {false, true, true}));
    }
    // Na - Mg: [1s2s2p][3s]
    for (int N = 11; N <= 12; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0}, {false, false, false, true}));
    }
    // Al - Ar: [1s2s2p][3s3p]
    for (int N = 13; N <= 18; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0, 1}, {false, false, false, true, true}));
    }
    // K  - Ca: [1s2s2p3s3p][4s]
    for (int N = 19; N <= 20; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0, 1, 0}, {false, false, false, false, false, true}));
    }
    // Sc - Zn: [1s2s2p3s3p4s][3d]
    for (int N = 21; N <= 30; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0, 1, 0, 2}, {false, false, false, false, false, false, true}));
    }
    // Ga - Kr: [1s2s2p3s3p3d][4s4p]
    for (int N = 31; N <= 36; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0, 1, 2, 0, 1}, {false, false, false, false, false, false, true, true}));
    }
    // Rb - Sr: [1s2s2p3s3p4s3d4p][5s]
    for (int N = 37; N <= 38; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0, 1, 0, 2, 1, 0}, {false, false, false, false, false, false, false, false, true}));
    }
    // Y  - Cd: [1s2s2p3s3p4s3d4p5s][4d]
    for (int N = 39; N <= 48; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0, 1, 0, 2, 1, 0, 2}, {false, false, false, false, false, false, false, false, false, true}));
    }
    // In - Xe: [1s2s3p3s3p4s3d4p4p][5s5p]
    for (int N = 49; N <= 54; N++) {
        atom_list.push_back(SADAtom(N, {0, 0, 1, 0, 1, 0, 2, 1, 2, 0, 1}, {false, false, false, false, false, false, false, false, false, true, true}));
    }

    // Idiot checks
    for (int N = 0; N < atom_list.size(); N++) {
        if (atom_list[N].N() != N) throw std::runtime_error("SAD Code broken - atom_list is malformed.");
        // SADAtom constructor checks lengths of Ls/acts
    }
    
    return atom_list;
}

std::string SADAtom::string() const
{
    std::string s;
    s += sprintf2("SADAtom: %d\n", N_);
    for (size_t ind = 0; ind < Ls_.size(); ind++) {
        s += sprintf2("  %1d: %s\n", Ls_[ind], (acts_[ind] ? "act" : "frz"));
    }
    return s;
}
std::string SADAtom::print_atoms()
{
    if (!atom_list__.size()) build_atom_list();
    std::string s;
    s += "SADAtom: Full Periodic Table:\n";
    for (size_t ind = 0; ind < atom_list__.size(); ind++) {
        s += "\n";
        s += atom_list__[ind].string();
    } 
    return s;
}

} // namespace lightspeed 
