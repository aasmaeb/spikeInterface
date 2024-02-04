from scipy.io import loadmat
import subprocess


# Set the path to the MATLAB executable
#data = loadmat(r"C:\Users\Hp\OneDrive\Bureau\PTA\example.mat")


#print(data)




chemin_fichier_matlab = r"C:\Users\USER\Documents\pta\fitLFPpowerSpectrum.m"
commande_matlab = f"matlab -r 'run(\"{chemin_fichier_matlab}\")'"

try:
    sortie = subprocess.check_output(commande_matlab, shell=True, stderr=subprocess.STDOUT, text=True)
    print("ca marche")
except subprocess.CalledProcessError as e:
    print(f"Erreur  : {e.output}")
    
    