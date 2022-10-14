/*

#############################################################################
#                                                                           #
#              Vibrational Heat-Bath Configuration Interaction              # 
#                         By: Jonathan H. Fetherolf                         #
#                                                                           #
#                      Based on LOVCI by Eric G. Kratz                      #
#                                                                           #
#############################################################################

 Implementation of Epstein-Nesbet PT2 with Heat Bath sorting

*/

void DoPT2(MatrixXd& Evecs, VectorXd& Evals){
    if(HCI_Eps==0){
        HeatBath_Sort_FC();
    }
    cout << " Finding connected states..." << endl;
    int N_opt;
    if(NEig > BasisSet.size()){ // If we don't have enough states to optimize for yet
        N_opt = BasisSet.size();
    }else{ // If number of states exceeds the number selected to optimize
        N_opt = NEig;
    }
    vector<double> Cmax; // Vector of maximum coefficients over excited states
        // MatrixXd absEvecs = abs(Evecs);
    for(unsigned int i=0; i<Evecs.rows(); i++){
        double Cij = Evecs(i,0);
        for(unsigned int j=1; j<N_opt; j++){
            if(abs(Evecs(i,j))>abs(Cij)){
                Cij = Evecs(i,j);
            }
        }
        Cmax.push_back(Cij);
    }
    HashedStates HashedBasisInit; // hashed unordered_set containing BasisSet to check for duplicates
    HashedStates HashedPTBasis; // hashed unordered_set of new states that only allows unique states to be inserted
    for(const WaveFunction& wfn : BasisSet){
        HashedBasisInit.insert(wfn); // Populate hashed unordered_set with initial basis states
    }
    for(unsigned int n=0; n<Cmax.size(); n++){ // Loop over max CI coefficients and add basis states
        AddStatesHB(HashedBasisInit,HashedPTBasis,n,Cmax[n],PT2_Eps);
    }
    int PTBasisSize = HashedPTBasis.size();
    cout << " Perturbative space contains " << PTBasisSize << " states." << endl;
    for(const WaveFunction& wfn : HashedPTBasis){ // Add NewStates to the BasisSet
        BasisSet.push_back(wfn);
    }
    HashedPTBasis = HashedStates(); // Free from memory
    HashedBasisInit = HashedStates(); // Free from memory
    //// PT Basis has been constructed; now we calculate matrix elements ////
    if(NEig==1){
        cout << " Calculating the 2nd-order perturbative correction on the ground state energy" << endl;
    }else{
        cout << " Calculating the 2nd-order perturbative correction for on first " << N_opt << " eigenvalues." << endl;
    }
    vector<double> DeltaE(N_opt,0.);  // Vector will contain the PT correction for each eigenvalue
    int fcmax=0;
    for (unsigned int k=0;k<AnharmFC.size();k++){ //Order of maximum anharmonic term
        if(AnharmFC[k].fcpow.size()>fcmax){
            fcmax = AnharmFC[k].fcpow.size();
        }
    }  
    #pragma omp parallel for
    for(unsigned int a=Evecs.rows(); a<Evecs.rows()+PTBasisSize;a++){
        vector<double> SumHaiCi(N_opt,0.);
        vector<int> qdiffvec(BasisSet[0].M,0);
        for(unsigned int i=0; i<Evecs.rows(); i++){
            double Hai = 0;
            int mchange = 0; // number of modes with nonzero change in quanta
            int qdiff = 0; // total number of quanta difference 
            QDiffVec(a,i,qdiff,mchange,qdiffvec);
            if(qdiff <= fcmax && mchange <= fcmax && qdiff%2==0){ 
                // States cannot differ by more than fcmax quanta
                for (unsigned int k=0;k<QuarticFC.size();k++){
                    if ( ScreenState(qdiff,mchange,qdiffvec,QuarticFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,QuarticFC[k]);
                    }
                }
                for (unsigned int k=0;k<SexticFC.size();k++){
                    if ( ScreenState(qdiff,mchange,qdiffvec,SexticFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,SexticFC[k]);
                    }
                }
                for(int n=0; n<N_opt; n++){
               //     #pragma omp critical
                    SumHaiCi[n] += Hai*Evecs(i,n); // C_i Hai for each eigenvalue of interest
                }
            }
            if(qdiff <= fcmax-1 && mchange <= fcmax-1 && qdiff%2==1){ 
                // fcmax-1 assumes 4th or 6th max order 
                // States cannot differ by more than fcmax quanta
                for (unsigned int k=0;k<CubicFC.size();k++){
                    if ( ScreenState(qdiff,mchange,qdiffvec,CubicFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,CubicFC[k]);
                    }
                }
                for (unsigned int k=0;k<QuinticFC.size();k++){
                    if ( ScreenState(qdiff,mchange,qdiffvec,QuinticFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,QuinticFC[k]);
                    }
                }
                for(int n=0; n<N_opt; n++){
               //     #pragma omp critical
                    SumHaiCi[n] += Hai*Evecs(i,n); // C_i Hai for each eigenvalue of interest
                }
            }
        }
        double Ea = 0.; //Hii matrix element
        for (unsigned int j=0;j<BasisSet[a].M;j++){
          //Calculate partial energies
          double Ej = 0.5;
          Ej += BasisSet[a].Modes[j].Quanta;
          Ej *= BasisSet[a].Modes[j].Freq;
          //Update matrix element
          Ea += Ej;
        }
        vector<int> zerodiffvec(BasisSet[0].M,0);
        int qdiff=0;
        int mchange=0;
        for (unsigned int k=0;k<QuarticFC.size();k++){ // Only even-ordered fc can affect this
            if ( ScreenState(qdiff,mchange,zerodiffvec,QuarticFC[k]) ){
            // Screen force constants that cannot connect basis states a and a
            //Add anharmonic matrix elements
                Ea += AnharmPot(a,a,QuarticFC[k]);
            }
        }
        for (unsigned int k=0;k<SexticFC.size();k++){ // Only even-ordered fc can affect this
            if ( ScreenState(qdiff,mchange,zerodiffvec,SexticFC[k]) ){
            // Screen force constants that cannot connect basis states a and a
            //Add anharmonic matrix elements
                Ea += AnharmPot(a,a,SexticFC[k]);
            }
        }
        for(int n=0; n<N_opt; n++){
            #pragma omp atomic // Will cause floating point error if blindly done in parallel
            DeltaE[n] += pow(SumHaiCi[n],2)/(Evals(n)-Ea);
        }
    }
    for(int n=0; n<N_opt; n++){
        Evals(n) += DeltaE[n];
        //std::cout << DeltaE[n] << std::endl;
    }
    cout << " New ZPE after PT2 correction is: " << Evals(0) << endl;
}

std::vector<double> StateProbability(std::vector<double>& CMax)
{
    double CTotal = 0.0;
    for (unsigned int i = 0; i < CMax.size(); i++)
    {
        CTotal += abs(CMax[i]);
    }

    std::vector<double> CProbability;
    for (int i = 0; i < CMax.size(); i++)
    {
        CProbability.push_back(abs(CMax[i]) / CTotal);
    }
    return CProbability;
}

void FillWalkers(std::vector<int>& WalkerPopulation, std::vector<double>& WalkerProbability, std::vector<double>& CMax, int Nd)
{
    WalkerProbability = StateProbability(CMax);
    std::vector<double> WalkerSummedProb;
    WalkerPopulation.clear();
    for (int i = 0; i < WalkerProbability.size(); i++)
    {
        if (i == 0) WalkerSummedProb.push_back(WalkerProbability[0]);
        else
        {
            WalkerSummedProb.push_back(WalkerSummedProb[i - 1] + WalkerProbability[i]);
        }
        WalkerPopulation.push_back(0);
    }

    for (int i = 0; i < Nd; i++)
    {
        double P = (double)std::rand() / (double) RAND_MAX;
        for (int w = 0; w < WalkerProbability.size(); w++)
        {
            double LB, UB;
            if (w == 0) LB = 0;
            else LB = WalkerSummedProb[w - 1];
            UB = WalkerSummedProb[w];
            if (P < UB && P > LB)
            {
                WalkerPopulation[w]++;
                break;
            }
        }
    }
}

std::vector<double> DoStocasticPT2(MatrixXd& Evecs, VectorXd& Evals, int Nd, double Epsilon3)
{
    cout << " Starting Stocastic PT2 corrections." << endl;
    if(HCI_Eps==0){
        HeatBath_Sort_FC();
    }
    cout << " Finding connected states..." << endl;
    int N_opt;
    if(NEig > BasisSet.size()){ // If we don't have enough states to optimize for yet
        N_opt = BasisSet.size();
    }else{ // If number of states exceeds the number selected to optimize
        N_opt = NEig;
    }
    vector<double> Cmax; // Vector of maximum coefficients over excited states
        // MatrixXd absEvecs = abs(Evecs);
    for(unsigned int i=0; i<Evecs.rows(); i++){
        double Cij = Evecs(i,0);
        for(unsigned int j=1; j<N_opt; j++){
            if(abs(Evecs(i,j))>abs(Cij)){
                Cij = Evecs(i,j);
            }
        }
        Cmax.push_back(Cij);
    }
    HashedStates HashedBasisInit; // hashed unordered_set containing BasisSet to check for duplicates
    HashedStates HashedPTBasis; // hashed unordered_set of new states that only allows unique states to be inserted
    for(const WaveFunction& wfn : BasisSet){
        HashedBasisInit.insert(wfn); // Populate hashed unordered_set with initial basis states
    }
    for(unsigned int n=0; n<Cmax.size(); n++){ // Loop over max CI coefficients and add basis states
        AddStatesHB(HashedBasisInit,HashedPTBasis,n,Cmax[n],Epsilon3);
    }
    int PTBasisSize = HashedPTBasis.size();
    cout << " Perturbative space contains " << PTBasisSize << " states." << endl;
    for(const WaveFunction& wfn : HashedPTBasis){ // Add NewStates to the BasisSet
        BasisSet.push_back(wfn);
    }
    HashedPTBasis = HashedStates(); // Free from memory
    HashedBasisInit = HashedStates(); // Free from memory
    //// PT Basis has been constructed; now we calculate matrix elements ////
    if(NEig==1){
        cout << " Calculating the 2nd-order perturbative correction on the ground state energy" << endl;
    }else{
        cout << " Calculating the 2nd-order perturbative correction for on first " << N_opt << " eigenvalues." << endl;
    }
    vector<double> DeltaE(N_opt,0.);  // Vector will contain the PT correction for each eigenvalue
    int fcmax=0;
    for (unsigned int k=0;k<AnharmFC.size();k++){ //Order of maximum anharmonic term
        if(AnharmFC[k].fcpow.size()>fcmax){
            fcmax = AnharmFC[k].fcpow.size();
        }
    }

    // This is the stocastic step - this first part is state specific
    /*
    std::vector<std::vector<double>> WalkerProbability; // Probability for each state
    std::vector<std::vector<int>> WalkerPopulation; // Population for each state
    for (int n = 0; n < N_opt; n++)
    {
        std::vector<double> WalkerProbN;
        std::vector<int> WalkerPopN;
        FillWalkers(WalkerPopN, WalkerProbN, Evecs.col(n), Nd);
        WalkerProbability.push_back(WalkerProbN);
        WalkerPopulation.push_back(WalkerPopN);
    }
    // Lets form a new C matrix that only includes populated states.
    std::vector<std::vector<double>> CWalker; // i, n
    for (int i = 0; i < Evecs.rows(); i++)
    {
        std::vector<double> Ci;
        for (int n = 0; n < N_opt; n++)
        {
            if (WalkerPopulation[n][i] != 0) Ci.push_back(Evecs(i, n) * WalkerPopulation[n][i]);
        }
        CWalker.push_back(Ci)
    }
    */

    std::vector<double> WalkerProbability;
    std::vector<int> WalkerPopulation;
    FillWalkers(WalkerPopulation, WalkerProbability, Cmax, Nd);
    
    #pragma omp parallel for
    for (unsigned int a=Evecs.rows(); a<Evecs.rows()+PTBasisSize;a++)
    {
        vector<double> SumHaiCi(N_opt,0.);
        std::vector<double> SumHaiCi2(N_opt, 0.);
        vector<int> qdiffvec(BasisSet[0].M,0);
        for (unsigned int i=0; i<Evecs.rows(); i++)
        {
            if (WalkerPopulation[i] == 0) continue;
            double Hai = 0;
            int mchange = 0; // number of modes with nonzero change in quanta
            int qdiff = 0; // total number of quanta difference 
            QDiffVec(a,i,qdiff,mchange,qdiffvec);
            if(qdiff <= fcmax && mchange <= fcmax && qdiff%2==0)
            { 
                // States cannot differ by more than fcmax quanta
                for (unsigned int k=0;k<QuarticFC.size();k++)
                {
                    if ( ScreenState(qdiff,mchange,qdiffvec,QuarticFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,QuarticFC[k]);
                    }
                }
                for (unsigned int k=0;k<SexticFC.size();k++)
                {
                    if ( ScreenState(qdiff,mchange,qdiffvec,SexticFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,SexticFC[k]);
                    }
                }
                for (int n=0; n<N_opt; n++)
                {
                //     #pragma omp critical
                    SumHaiCi[n] += (Hai*Evecs(i,n) * WalkerPopulation[i]) / WalkerProbability[i]; // C_i Hai for each eigenvalue of interest
                    SumHaiCi2[n] += (pow(Hai, 2) * pow(Evecs(i, n), 2)) * (WalkerPopulation[i] * (Nd - 1) / WalkerProbability[i] - pow(WalkerPopulation[i], 2) / pow(WalkerProbability[i], 2));
                }
            }
            if(qdiff <= fcmax-1 && mchange <= fcmax-1 && qdiff%2==1)
            { 
                // fcmax-1 assumes 4th or 6th max order 
                // States cannot differ by more than fcmax quanta
                for (unsigned int k=0;k<CubicFC.size();k++)
                {
                    if ( ScreenState(qdiff,mchange,qdiffvec,CubicFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,CubicFC[k]);
                    }
                }
                for (unsigned int k=0;k<QuinticFC.size();k++)
                {
                    if ( ScreenState(qdiff,mchange,qdiffvec,QuinticFC[k]) ){
                        //Screen force constants for connection                      
                        //Add anharmonic matrix elements
                        Hai += AnharmPot(a,i,QuinticFC[k]);
                    }
                }
                for(int n=0; n<N_opt; n++)
                {
                //     #pragma omp critical
                    SumHaiCi[n] += (Hai*Evecs(i,n) * WalkerPopulation[i]) / WalkerProbability[i]; // C_i Hai for each eigenvalue of interest
                    SumHaiCi2[n] += (pow(Hai, 2) * pow(Evecs(i, n), 2)) * (WalkerPopulation[i] * (Nd - 1) / WalkerProbability[i] - pow(WalkerPopulation[i], 2) / pow(WalkerProbability[i], 2));
                }
            }
        }
        double Ea = 0.; //Hii matrix element
        for (unsigned int j=0;j<BasisSet[a].M;j++){
          //Calculate partial energies
          double Ej = 0.5;
          Ej += BasisSet[a].Modes[j].Quanta;
          Ej *= BasisSet[a].Modes[j].Freq;
          //Update matrix element
          Ea += Ej;
        }
        vector<int> zerodiffvec(BasisSet[0].M,0);
        int qdiff=0;
        int mchange=0;
        for (unsigned int k=0;k<QuarticFC.size();k++){ // Only even-ordered fc can affect this
            if ( ScreenState(qdiff,mchange,zerodiffvec,QuarticFC[k]) ){
            // Screen force constants that cannot connect basis states a and a
            //Add anharmonic matrix elements
                Ea += AnharmPot(a,a,QuarticFC[k]);
            }
        }
        for (unsigned int k=0;k<SexticFC.size();k++){ // Only even-ordered fc can affect this
            if ( ScreenState(qdiff,mchange,zerodiffvec,SexticFC[k]) ){
            // Screen force constants that cannot connect basis states a and a
            //Add anharmonic matrix elements
                Ea += AnharmPot(a,a,SexticFC[k]);
            }
        }
        for(int n=0; n<N_opt; n++){
            #pragma omp atomic // Will cause floating point error if blindly done in parallel
            DeltaE[n] += (pow(SumHaiCi[n],2) + SumHaiCi2[n]) / ((Evals(n)-Ea) * Nd * (Nd - 1));
        }
    }
    /*
    for(int n=0; n<N_opt; n++){
        Evals(n) += DeltaE[n];
        std::cout << DeltaE[n] << std::endl;
    }
    cout << " New ZPE after PT2 correction is: " << Evals(0) << endl;
    */
    return DeltaE;
}

void DoPT2_StateSpecific(MatrixXd& Evecs, VectorXd& Evals)
{
    cout << " Starting PT2 corrections." << endl;
    if(HCI_Eps==0){
        HeatBath_Sort_FC();
    }
    cout << " Finding connected states..." << endl;
    int N_opt;
    if(NEig > BasisSet.size()){ // If we don't have enough states to optimize for yet
        N_opt = BasisSet.size();
    }else{ // If number of states exceeds the number selected to optimize
        N_opt = NEig;
    }

    if(NEig==1){
        cout << " Calculating the 2nd-order perturbative correction on the ground state energy" << endl;
    }else{
        cout << " Calculating the 2nd-order perturbative correction for on first " << N_opt << " eigenvalues." << endl;
    }
 
    vector<double> DeltaE(N_opt,0.);  // Vector will contain the PT correction for each eigenvalue
    int fcmax=0;
    for (unsigned int k=0;k<AnharmFC.size();k++){ //Order of maximum anharmonic term
        if(AnharmFC[k].fcpow.size()>fcmax){
            fcmax = AnharmFC[k].fcpow.size();
        }
    }

    HashedStates HashedBasisInit; // hashed unordered_set containing BasisSet to check for duplicates
    for(const WaveFunction& wfn : BasisSet){
        HashedBasisInit.insert(wfn); // Populate hashed unordered_set with initial basis states
    }
    std::vector<WaveFunction> Basis0 = BasisSet; 

    for (unsigned int n = 0; n < N_opt; n++)
    {
        // For each state, we determine the perturbative basis and populate the state with walkers.
        
        HashedStates HashedPTBasis;
        std::vector<double> Cn;
        for (unsigned int i = 0; i < Evecs.rows(); i++) 
        {
            AddStatesHB(HashedBasisInit, HashedPTBasis, i, Evecs(i, n), PT2_Eps);
            Cn.push_back(Evecs(i, n));
        }
        int PTBasisSize = HashedPTBasis.size();
        cout << " Perturbative space for state " << n << " contains " << PTBasisSize << " states." << endl;
        BasisSet = Basis0;
        for (const WaveFunction &B : HashedPTBasis) BasisSet.push_back(B);

        #pragma omp parallel for
        for (unsigned int a = Evecs.rows(); a < Evecs.rows() + PTBasisSize; a++)
        {
            double HaiCi = 0.0;
            vector<int> qdiffvec(BasisSet[0].M,0);
            for (unsigned int i = 0; i < Evecs.rows(); i++)
            {
                double Hai = 0;
                int mchange = 0; // number of modes with nonzero change in quanta
                int qdiff = 0; // total number of quanta difference 
                QDiffVec(a,i,qdiff,mchange,qdiffvec);
                if(qdiff <= fcmax && mchange <= fcmax && qdiff%2==0)
                { 
                    // States cannot differ by more than fcmax quanta
                    for (unsigned int k=0;k<QuarticFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,QuarticFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,QuarticFC[k]);
                        }
                    }
                    for (unsigned int k=0;k<SexticFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,SexticFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,SexticFC[k]);
                        }
                    }
                    HaiCi += Hai * Evecs(i, n); // C_i Hai for each eigenvalue of interest
                }
                if(qdiff <= fcmax-1 && mchange <= fcmax-1 && qdiff%2==1)
                { 
                    // fcmax-1 assumes 4th or 6th max order 
                    // States cannot differ by more than fcmax quanta
                    for (unsigned int k=0;k<CubicFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,CubicFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,CubicFC[k]);
                        }
                    }
                    for (unsigned int k=0;k<QuinticFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,QuinticFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,QuinticFC[k]);
                        }
                    }
                    HaiCi += Hai * Evecs(i, n); // C_i Hai for each eigenvalue of interest
                }
            }
            double Ea = 0.; //Hii matrix element
            for (unsigned int j = 0; j < BasisSet[a].M; j++)
            {
              //Calculate partial energies
              double Ej = 0.5;
              Ej += BasisSet[a].Modes[j].Quanta;
              Ej *= BasisSet[a].Modes[j].Freq;
              //Update matrix element
              Ea += Ej;
            }
            vector<int> zerodiffvec(BasisSet[0].M,0);
            int qdiff=0;
            int mchange=0;
            for (unsigned int k = 0; k < QuarticFC.size(); k++) // Only even-ordered fc can affect this
            {
                if (ScreenState(qdiff, mchange, zerodiffvec, QuarticFC[k]))
                {
                    // Screen force constants that cannot connect basis states a and a
                    //Add anharmonic matrix elements
                    Ea += AnharmPot(a,a,QuarticFC[k]);
                }
            }
            for (unsigned int k = 0; k < SexticFC.size(); k++) // Only even-ordered fc can affect this
            {    
                if (ScreenState(qdiff, mchange, zerodiffvec, SexticFC[k]))
                {
                    // Screen force constants that cannot connect basis states a and a
                    //Add anharmonic matrix elements
                    Ea += AnharmPot(a,a,SexticFC[k]);
                }
            }
            #pragma omp atomic // Will cause floating point error if blindly done in parallel
            DeltaE[n] += pow(HaiCi, 2) / (Evals(n) - Ea);
        }
    }
    
    for (int n = 0; n < N_opt; n++)
    {
        Evals(n) += DeltaE[n];
        //std::cout << DeltaE[n] << std::endl;
    }
    cout << " New ZPE after PT2 correction is: " << Evals(0) << endl;
}

std::vector<double> DoStocasticPT2_StateSpecific(MatrixXd& Evecs, VectorXd& Evals, int Nd, double Epsilon3)
{
    cout << " Starting Stocastic PT2 corrections." << endl;
    if(HCI_Eps==0){
        HeatBath_Sort_FC();
    }
    cout << " Finding connected states..." << endl;
    int N_opt;
    if(NEig > BasisSet.size()){ // If we don't have enough states to optimize for yet
        N_opt = BasisSet.size();
    }else{ // If number of states exceeds the number selected to optimize
        N_opt = NEig;
    }

    if(NEig==1){
        cout << " Calculating the 2nd-order perturbative correction on the ground state energy" << endl;
    }else{
        cout << " Calculating the 2nd-order perturbative correction for on first " << N_opt << " eigenvalues." << endl;
    }
 
    vector<double> DeltaE(N_opt,0.);  // Vector will contain the PT correction for each eigenvalue
    int fcmax=0;
    for (unsigned int k=0;k<AnharmFC.size();k++){ //Order of maximum anharmonic term
        if(AnharmFC[k].fcpow.size()>fcmax){
            fcmax = AnharmFC[k].fcpow.size();
        }
    }

    HashedStates HashedBasisInit; // hashed unordered_set containing BasisSet to check for duplicates
    for(const WaveFunction& wfn : BasisSet){
        HashedBasisInit.insert(wfn); // Populate hashed unordered_set with initial basis states
    }
    std::vector<WaveFunction> Basis0 = BasisSet; 

    for (unsigned int n = 0; n < N_opt; n++)
    {
        // For each state, we determine the perturbative basis and populate the state with walkers.
        
        HashedStates HashedPTBasis;
        std::vector<double> Cn;
        for (unsigned int i = 0; i < Evecs.rows(); i++) 
        {
            AddStatesHB(HashedBasisInit, HashedPTBasis, i, Evecs(i, n), Epsilon3);
            Cn.push_back(Evecs(i, n));
        }
        int PTBasisSize = HashedPTBasis.size();
        cout << " Perturbative space for state " << n << " contains " << PTBasisSize << " states." << endl;
        BasisSet = Basis0;
        for (const WaveFunction &B : HashedPTBasis) BasisSet.push_back(B);

        std::vector<double> WalkerProbability;
        std::vector<int> WalkerPopulation;
        FillWalkers(WalkerPopulation, WalkerProbability, Cn, Nd);

        #pragma omp parallel for
        for (unsigned int a = Evecs.rows(); a < Evecs.rows() + PTBasisSize; a++)
        {
            double HaiCi = 0.0;
            double Hai2Ci2 = 0.0;
            vector<int> qdiffvec(BasisSet[0].M,0);
            for (unsigned int i = 0; i < Evecs.rows(); i++)
            {
                if (WalkerPopulation[i] == 0) continue;
                double Hai = 0;
                int mchange = 0; // number of modes with nonzero change in quanta
                int qdiff = 0; // total number of quanta difference 
                QDiffVec(a,i,qdiff,mchange,qdiffvec);
                if(qdiff <= fcmax && mchange <= fcmax && qdiff%2==0)
                { 
                    // States cannot differ by more than fcmax quanta
                    for (unsigned int k=0;k<QuarticFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,QuarticFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,QuarticFC[k]);
                        }
                    }
                    for (unsigned int k=0;k<SexticFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,SexticFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,SexticFC[k]);
                        }
                    }
                    HaiCi += (Hai * Evecs(i, n) * WalkerPopulation[i]) / WalkerProbability[i]; // C_i Hai for each eigenvalue of interest
                    Hai2Ci2 += (pow(Hai, 2) * pow(Evecs(i, n), 2)) * (WalkerPopulation[i] * (Nd - 1) / WalkerProbability[i] - pow(WalkerPopulation[i], 2) / pow(WalkerProbability[i], 2));
                }
                if(qdiff <= fcmax-1 && mchange <= fcmax-1 && qdiff%2==1)
                { 
                    // fcmax-1 assumes 4th or 6th max order 
                    // States cannot differ by more than fcmax quanta
                    for (unsigned int k=0;k<CubicFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,CubicFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,CubicFC[k]);
                        }
                    }
                    for (unsigned int k=0;k<QuinticFC.size();k++)
                    {
                        if ( ScreenState(qdiff,mchange,qdiffvec,QuinticFC[k]) ){
                            //Screen force constants for connection                      
                            //Add anharmonic matrix elements
                            Hai += AnharmPot(a,i,QuinticFC[k]);
                        }
                    }
                    HaiCi += (Hai * Evecs(i, n) * WalkerPopulation[i]) / WalkerProbability[i]; // C_i Hai for each eigenvalue of interest
                    Hai2Ci2 += (pow(Hai, 2) * pow(Evecs(i, n), 2)) * (WalkerPopulation[i] * (Nd - 1) / WalkerProbability[i] - pow(WalkerPopulation[i], 2) / pow(WalkerProbability[i], 2));
                }
            }
            double Ea = 0.; //Hii matrix element
            for (unsigned int j = 0; j < BasisSet[a].M; j++)
            {
              //Calculate partial energies
              double Ej = 0.5;
              Ej += BasisSet[a].Modes[j].Quanta;
              Ej *= BasisSet[a].Modes[j].Freq;
              //Update matrix element
              Ea += Ej;
            }
            vector<int> zerodiffvec(BasisSet[0].M,0);
            int qdiff=0;
            int mchange=0;
            for (unsigned int k = 0; k < QuarticFC.size(); k++) // Only even-ordered fc can affect this
            {
                if (ScreenState(qdiff, mchange, zerodiffvec, QuarticFC[k]))
                {
                    // Screen force constants that cannot connect basis states a and a
                    //Add anharmonic matrix elements
                    Ea += AnharmPot(a,a,QuarticFC[k]);
                }
            }
            for (unsigned int k = 0; k < SexticFC.size(); k++) // Only even-ordered fc can affect this
            {    
                if (ScreenState(qdiff, mchange, zerodiffvec, SexticFC[k]))
                {
                    // Screen force constants that cannot connect basis states a and a
                    //Add anharmonic matrix elements
                    Ea += AnharmPot(a,a,SexticFC[k]);
                }
            }
            #pragma omp atomic // Will cause floating point error if blindly done in parallel
            DeltaE[n] += (pow(HaiCi, 2) + Hai2Ci2) / ((Evals(n) - Ea) * Nd * (Nd - 1));
        }
    }
    
    /*
    for (int n = 0; n < N_opt; n++)
    {
        Evals(n) += DeltaE[n];
        std::cout << DeltaE[n] << std::endl;
    }
    cout << " New ZPE after PT2 correction is: " << Evals(0) << endl;
    */
    return DeltaE;
}


void DoSSPT2(MatrixXd& Evecs, VectorXd& Evals)
{
    DoPT2_StateSpecific(Evecs, Evals);
    std::vector<double> dE_Loose = DoStocasticPT2_StateSpecific(Evecs, Evals, NWalkers, PT2_Eps);
    std::vector<double> dE_Tight = DoStocasticPT2_StateSpecific(Evecs, Evals, NWalkers, SPT2_Eps);
    for (unsigned int n = 0; n < dE_Loose.size(); n++)
    {
        Evals[n] += (dE_Tight[n] - dE_Loose[n]);
        //std::cout << dE_Tight[n] - dE_Loose[n] << std::endl;
    }
}
