#include <lightspeed/molden.hpp>
#include <lightspeed/tensor.hpp>
#include <lightspeed/molecule.hpp>
#include <lightspeed/basis.hpp>
#include <cstdio>
#include <stdexcept>

namespace lightspeed {

void Molden::save_molden_file(
    const std::string& filename,
    const std::shared_ptr<Molecule>& molecule,
    const std::shared_ptr<Basis>& basis,
    const std::shared_ptr<Tensor>& C,
    const std::shared_ptr<Tensor>& eps,
    const std::shared_ptr<Tensor>& occ,
    bool alpha)
{
    size_t no = C->shape()[1];
    
    // Error checks
    C->shape_error({basis->nao(),no});
    eps->shape_error({no});
    occ->shape_error({no});

    if (basis->max_L() > 4) {
        throw std::runtime_error("Molden: Molden format only supports up to g functions");
    }
    bool cart = false;
    bool pure = false;
    for (auto shell : basis->shells()) {
        if (shell.is_pure()) pure = true;
        if (!shell.is_pure()) cart = true;;
    }
    if (cart && pure) {
        throw std::runtime_error("Molden: Molden cannot support mixed pure and cartesian functions");
    }
    
    // Open file
    FILE* fh = fopen(filename.c_str(),"w");
    if (fh == NULL) {
        throw std::runtime_error("Molden: file not found: " + filename);
    } 

    // Molden Header
    fprintf(fh, "[Molden Format]\n");
    fprintf(fh, "[Title]\n");
    fprintf(fh, "Written by Lightspeed\n");

    // Molecule
    fprintf(fh, "[Atoms] AU\n");
    for (size_t A = 0; A < molecule->natom(); A++) {
        const Atom& atom = molecule->atoms()[A];
        fprintf(fh, "%-2s %6zu %3d %14.6f %14.6f %14.6f\n",
            atom.symbol().c_str(),
            A + 1,
            atom.N(),
            atom.x(),
            atom.y(),
            atom.z());
    }

    std::vector<std::vector<size_t> > atom_inds(molecule->natom());
    for (size_t P =0; P < basis->shells().size(); P++) {
        atom_inds[basis->shells()[P].atomIdx()].push_back(P); 
    }

    // Basis Set
    fprintf(fh, "[GTO]\n");
    std::vector<char> shell_labels = { 's', 'p', 'd', 'f', 'g' };
    for (size_t A = 0; A < molecule->natom(); A++) {
        fprintf(fh, "%-6zu 0\n", A+1); // != 0 is no longer used
        for (size_t P : atom_inds[A]) {
            const Shell& shell = basis->shells()[P];
            // != 1.00 is no longer used
            fprintf(fh, "%c %-2d 1.00\n", shell_labels[shell.L()], shell.nprim()); 
            for (size_t K = 0; K < shell.nprim(); K++) {
                fprintf(fh, "%15.7f %15.7f\n", 
                    shell.es()[K],
                    shell.c0s()[K]); 
            }
        } 
        fprintf(fh, "\n");
    }
    
    // Puream specification
    if (pure) {
        fprintf(fh,"[5D]\n"); // Means 5D/7F
        fprintf(fh,"[9G]\n"); // Means 9G
    }

    // TODO: Check d and higher output in Cartesian, may need CCA normalization factor

    // Orbitals
    fprintf(fh, "[MO]\n");
    std::vector<std::vector<int>> pure_perm = {
        { 0 },
        { 1, 2, 0 }, // p is always x, y, z in molden
        { 0, 1, 2, 3, 4},
        { 0, 1, 2, 3, 4, 5, 6},
        { 0, 1, 2, 3, 4, 5, 6, 7, 8} };
    std::vector<std::vector<int>> cart_perm = {
        { 0 },
        { 0, 1, 2 },
        { 0, 3, 5, 1, 2, 4}, 
        { 0, 6, 9, 3, 1, 2, 5, 8, 7, 4},
        { 0, 10, 14, 1, 2, 6, 11, 9, 13, 3, 5, 12, 4, 7, 8} }; 
    const std::vector<double>& Cp = C->data();
    const std::vector<double>& epsp = eps->data();
    const std::vector<double>& occp = occ->data();
    for (size_t i = 0; i < no; i++) {
        fprintf(fh," Ene= %.6f\n", epsp[i]);
        fprintf(fh," Spin= %s\n", (alpha ? "Alpha" : "Beta"));
        fprintf(fh," Occup= %.6f\n", occp[i]);
        for (auto shell : basis->shells()) {
            size_t oP = shell.aoIdx();
            size_t nP = shell.nao();
            int L = shell.L();
            if (pure) {
                // Fortunately, already in the correct order (except p)
                for (size_t p = 0; p < nP; p++) {
                    size_t p2 = pure_perm[L][p];
                    fprintf(fh, " %-6zu %15.7f\n", p + oP + 1, Cp[(p2 + oP) * no + i]);
                }
            } else {
                // Molden likes Alexical ordering
                for (size_t p = 0; p < nP; p++) {
                    size_t p2 = cart_perm[L][p];
                    fprintf(fh, " %-6zu %15.7f\n", p + oP + 1, Cp[(p2 + oP) * no + i]);
                }
            }
        }
    }

    // Close File
    fclose(fh);
}

} // namespace lightspeed


