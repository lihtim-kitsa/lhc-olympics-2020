#include <iostream>
#include <TFile.h>
#include <TH1D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TMath.h>

// Dijet function defined by: p0 * (1 - x)^p1 / (x^(p2 + p3*log(x)))
// where x = mjj / 13000
Double_t dijet_func(Double_t *x, Double_t *par) {
    Double_t mjj = x[0];
    Double_t x_scaled = mjj / 13000.0;
    Double_t p0 = par[0];
    Double_t p1 = par[1];
    Double_t p2 = par[2];
    Double_t p3 = par[3];
    
    if (x_scaled >= 1.0) return 0.0;
    
    return p0 * TMath::Power(1.0 - x_scaled, p1) / TMath::Power(x_scaled, p2 + p3 * TMath::Log(x_scaled));
}

// Sideband rejection function for fitting
Double_t dijet_func_sideband(Double_t *x, Double_t *par) {
    Double_t mjj = x[0];
    // Reject signal region (3.3 TeV to 3.7 TeV)
    if (mjj > 3300.0 && mjj < 3700.0) {
        TF1::RejectPoint();
        return 0;
    }
    return dijet_func(x, par);
}

void bump_hunt(const char* root_file_path = "reports/M1_Autoencoder_42_spectra.root") {
    gStyle->SetOptFit(1111);
    
    TFile *f = TFile::Open(root_file_path);
    if (!f || f->IsZombie()) {
        std::cerr << "Error opening file: " << root_file_path << std::endl;
        return;
    }
    
    TH1D *h_incl = (TH1D*)f->Get("inclusive");
    if (!h_incl) {
        std::cerr << "Histogram 'inclusive' not found!" << std::endl;
        return;
    }
    
    // Create the TF1 for sideband fitting
    TF1 *fit_sideband = new TF1("fit_sideband", dijet_func_sideband, 2500, 4500, 4);
    fit_sideband->SetParameters(h_incl->Integral() * 10, 5.0, 5.0, 0.0);
    fit_sideband->SetParNames("N", "p1", "p2", "p3");
    
    // Fit the histogram
    std::cout << "Fitting inclusive spectrum..." << std::endl;
    h_incl->Fit("fit_sideband", "R");
    
    // Extrapolate to signal region
    TF1 *fit_full = new TF1("fit_full", dijet_func, 2500, 4500, 4);
    fit_full->SetParameters(fit_sideband->GetParameters());
    fit_full->SetLineColor(kBlue);
    fit_full->SetLineStyle(2);
    
    // Calculate expected background in signal region
    Double_t expected_bkg = fit_full->Integral(3300, 3700) / h_incl->GetBinWidth(1);
    
    // Calculate observed events in signal region
    int bin1 = h_incl->FindBin(3300);
    int bin2 = h_incl->FindBin(3700) - 1;
    Double_t observed = h_incl->Integral(bin1, bin2);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Signal Region: [3300, 3700] GeV" << std::endl;
    std::cout << "Observed Events: " << observed << std::endl;
    std::cout << "Expected Background: " << expected_bkg << std::endl;
    
    if (observed > expected_bkg && expected_bkg > 0) {
        Double_t z = TMath::Sqrt(2 * (observed * TMath::Log(observed / expected_bkg) - (observed - expected_bkg)));
        std::cout << "Local Significance (Z): " << z << " sigma" << std::endl;
    } else {
        std::cout << "No excess observed." << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    // Draw
    TCanvas *c1 = new TCanvas("c1", "Bump Hunt", 800, 600);
    h_incl->Draw("E");
    fit_full->Draw("SAME");
    c1->SaveAs("reports/bump_hunt_cpp.png");
}

int main(int argc, char** argv) {
    if (argc > 1) {
        bump_hunt(argv[1]);
    } else {
        bump_hunt();
    }
    return 0;
}
