#include <lightspeed/am.hpp>
#include <cmath>
#include <tuple>

namespace lightspeed {

AngularMomentum::AngularMomentum(int L) :
    L_(L)
{
    build_cart();
    build_pure();
}
void AngularMomentum::build_cart()
{
    ls_.resize(ncart(),0); 
    ms_.resize(ncart(),0); 
    ns_.resize(ncart(),0); 

    for (int i=0, index = 0; i <= L_; ++i) {
        int l = L_-i;
        for (int j=0; j<=i; ++j, ++index) {
            int m = i-j;
            int n = j;
            ls_[index] = l;
            ms_[index] = m;
            ns_[index] = n;
        }
    }
}
std::vector<AngularMomentum> AngularMomentum::build(int Lmax)
{
    std::vector<AngularMomentum> val(Lmax+1L);
    for (int L = 0; L <= Lmax; L++) {
        val[L] = AngularMomentum(L);
    }
    return val;
}

/// Each element contains a component of the form coef, lx, ly, lz
typedef std::vector<std::tuple<double, int, int, int> > SH;

/// Sum equivalent lx, ly, lz contributions
SH clean_SH(const SH& sh1)
{
    SH sh2;
    for (size_t ind1 = 0; ind1 < sh1.size(); ind1++) {
        double coef1 = std::get<0>(sh1[ind1]);
        int lx1 = std::get<1>(sh1[ind1]);
        int ly1 = std::get<2>(sh1[ind1]);
        int lz1 = std::get<3>(sh1[ind1]);
        bool found = false;
        for (size_t ind2 = 0; ind2 < sh2.size(); ind2++) {
            double coef2 = std::get<0>(sh2[ind2]);
            int lx2 = std::get<1>(sh2[ind2]);
            int ly2 = std::get<2>(sh2[ind2]);
            int lz2 = std::get<3>(sh2[ind2]);
            if (lx1 == lx2 && ly1 == ly2 && lz1 == lz2) {
                sh2[ind2] = std::make_tuple(coef1 + coef2, lx2, ly2, lz2);
                found = true;
                break;
            }
        }
        if (!found) sh2.push_back(sh1[ind1]); 
    } 

    SH sh3;
    for (size_t ind1 = 0; ind1 < sh2.size(); ind1++) {
        if (fabs(std::get<0>(sh2[ind1])) > 1.0E-15) 
            sh3.push_back(sh2[ind1]);
    }

    return sh3;
}
void AngularMomentum::build_pure()
{
    // => Base Case <= //

    /**
     * S_0,+0 = 1 (Purple Book 6.4.70)
     * S_0,-0 = 0 (effectively)
     **/
    std::vector<SH> Sllc(L_ + 1);
    std::vector<SH> Slls(L_ + 1);
    Sllc[0].push_back(std::make_tuple(1.0,0,0,0));
    Slls[0].push_back(std::make_tuple(0.0,0,0,0));
 
    // => Diagonal Recursion <= //

    /**
     * S_l+1,+l+1 = sqrt{2^{\delta_{l0}} \frac{2l+1}{2l+2}} (x S_l,l - (1 - \delta_{l0}) y S_l,-l) (Purple Book 6.4.71)
     * S_l+1,-l-1 = sqrt{2^{\delta_{l0}} \frac{2l+1}{2l+2}} (y S_l,l + (1 - \delta_{l0}) x S_l,-l) (Purple Book 6.4.72)
     **/
    for (int l = 0; l < L_; l++) {
        double Nc = sqrt((l == 0 ? 2.0 : 1.0) * (2.0 * l + 1) / (2.0 * l + 2.0));
        double Ns = (l == 0 ? 0.0 : Nc);
        SH Sl1l1c;
        SH Sl1l1s;
        for (size_t ind = 0; ind < Sllc[l].size(); ind++) {
            double coef = std::get<0>(Sllc[l][ind]);
            int lx = std::get<1>(Sllc[l][ind]);
            int ly = std::get<2>(Sllc[l][ind]);
            int lz = std::get<3>(Sllc[l][ind]);
            Sl1l1c.push_back(std::make_tuple(Nc * coef, lx + 1, ly, lz));
            Sl1l1s.push_back(std::make_tuple(Nc * coef, lx, ly + 1, lz));
        }
        for (size_t ind = 0; ind < Slls[l].size(); ind++) {
            double coef = std::get<0>(Slls[l][ind]);
            int lx = std::get<1>(Slls[l][ind]);
            int ly = std::get<2>(Slls[l][ind]);
            int lz = std::get<3>(Slls[l][ind]);
            Sl1l1c.push_back(std::make_tuple(- Ns * coef, lx, ly + 1, lz));
            Sl1l1s.push_back(std::make_tuple(  Ns * coef, lx + 1, ly, lz));
        }
        
        Sllc[l+1] = clean_SH(Sl1l1c); 
        Slls[l+1] = clean_SH(Sl1l1s); 
    }     

    // => Leaf Recursion <= //

    /**
     * S_l+1,m = \frac{(2l+1) z S_l,m - sqrt{(l+m)(l-m)} r^2 S_l-1,m}{\sqrt{(l+m+1)(l-m+1)}} (Purple Book 6.7.73)
     **/
    std::vector<SH> SLm(2 * L_ + 1);

    for (int m = 0; m <= L_; m++) {
        SH Slmc = Sllc[m];
        SH Slms = Slls[m];
        SH Sldmc;
        SH Sldms;
        Sldmc.push_back(std::make_tuple(0.0,0,0,0));
        Sldms.push_back(std::make_tuple(0.0,0,0,0));
        for (int l = m; l < L_; l++) {
            double a = (2.0 * l + 1.0) / sqrt((l + m + 1.0) * (l - m + 1.0));
            double b = sqrt(((l + m) * (double)(l - m)) / ((l + m + 1.0) * (l - m + 1.0)));
            SH Slumc;
            SH Slums;    
            for (size_t ind = 0; ind < Slmc.size(); ind++) {
                double coef = std::get<0>(Slmc[ind]);
                int lx = std::get<1>(Slmc[ind]);
                int ly = std::get<2>(Slmc[ind]);
                int lz = std::get<3>(Slmc[ind]);
                Slumc.push_back(std::make_tuple(a * coef, lx, ly, lz + 1));
            }
            for (size_t ind = 0; ind < Slms.size(); ind++) {
                double coef = std::get<0>(Slms[ind]);
                int lx = std::get<1>(Slms[ind]);
                int ly = std::get<2>(Slms[ind]);
                int lz = std::get<3>(Slms[ind]);
                Slums.push_back(std::make_tuple(a * coef, lx, ly, lz + 1));
            }
            for (size_t ind = 0; ind < Sldmc.size(); ind++) {
                double coef = std::get<0>(Sldmc[ind]);
                int lx = std::get<1>(Sldmc[ind]);
                int ly = std::get<2>(Sldmc[ind]);
                int lz = std::get<3>(Sldmc[ind]);
                Slumc.push_back(std::make_tuple(-b * coef, lx + 2, ly, lz));
                Slumc.push_back(std::make_tuple(-b * coef, lx, ly + 2, lz));
                Slumc.push_back(std::make_tuple(-b * coef, lx, ly, lz + 2));
            }
            for (size_t ind = 0; ind < Sldms.size(); ind++) {
                double coef = std::get<0>(Sldms[ind]);
                int lx = std::get<1>(Sldms[ind]);
                int ly = std::get<2>(Sldms[ind]);
                int lz = std::get<3>(Sldms[ind]);
                Slums.push_back(std::make_tuple(-b * coef, lx + 2, ly, lz));
                Slums.push_back(std::make_tuple(-b * coef, lx, ly + 2, lz));
                Slums.push_back(std::make_tuple(-b * coef, lx, ly, lz + 2));
            }
            Slumc = clean_SH(Slumc);
            Slums = clean_SH(Slums);
            Sldmc = Slmc;
            Sldms = Slms;
            Slmc = Slumc;
            Slms = Slums;
        }
        if (m == 0) {
            SLm[0] = Slmc;
        } else {
            SLm[2 * m - 1] = Slmc;
            SLm[2 * m]     = Slms;
        }
    }
    
    // => Unpacking <= //

    for (size_t ind1 = 0; ind1 < SLm.size(); ind1++) {
        for (size_t ind2 = 0; ind2 < SLm[ind1].size(); ind2++) {
            double coef = std::get<0>(SLm[ind1][ind2]);
            int lx = std::get<1>(SLm[ind1][ind2]);
            int ly = std::get<2>(SLm[ind1][ind2]);
            int lz = std::get<3>(SLm[ind1][ind2]);
            for (size_t ind3 = 0; ind3 < ls_.size(); ind3++) {
                if (lx == ls_[ind3] && ly == ms_[ind3] && lz == ns_[ind3]) {
                    cart_inds_.push_back(ind3);
                    pure_inds_.push_back(ind1);
                    cart_coefs_.push_back(coef);
                    continue;
                }
            }
        }   
    }
}

} // namespace lightspeed
