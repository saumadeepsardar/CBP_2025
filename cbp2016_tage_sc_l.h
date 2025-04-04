#ifndef _TAGE_PREDICTOR_H_
#define _TAGE_PREDICTOR_H_

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <array>
#include <iostream>

// Configuration Parameters
#define UINT64 uint64_t
#define HISTBUFFERLENGTH 4096 // Global history buffer size
#define BORNTICK 1024         // Threshold for resetting usefulness counters
#define PRINTSIZE             // Enable storage size printing for debugging

// TAGE Parameters (adjusted to fit RL within budget)
#define NHIST 32                        // Number of TAGE tables (reduced from 36)
#define NBANKLOW 10                     // Low history length banks
#define NBANKHIGH 18                    // High history length banks (reduced from 20)
#define BORN 13                         // Boundary between low and high history lengths
#define BORNINFASSOC 9                  // Start of 2-way associativity
#define BORNSUPASSOC 23                 // End of 2-way associativity
#define MINHIST 6                       // Shortest history length
#define MAXHIST 3000                    // Longest history length
#define LOGG 10                         // Log size of tagged TAGE tables (1024 entries)
#define TBITS 8                         // Minimum tag width (8 for low, 12 for high)
#define NNN 1                           // Extra entries allocated on misprediction
#define HYSTSHIFT 2                     // Hysteresis shared by 4 bimodal entries
#define LOGB 13                         // Log size of bimodal table (8K entries)
#define PHISTWIDTH 27                   // Width of path history
#define UWIDTH 1                        // Usefulness counter width
#define CWIDTH 3                        // Prediction counter width
#define LOGSIZEUSEALT 4                 // Log size of alternate prediction confidence table
#define ALTWIDTH 5                      // Width of alternate prediction counter
#define SIZEUSEALT (1 << LOGSIZEUSEALT) // 16 entries

// Loop Predictor Parameters
#define LOOPPREDICTOR      // Enable loop predictor
#define LOGL 5             // Log size of loop table (32 entries)
#define WIDTHNBITERLOOP 10 // Max 1K iterations
#define LOOPTAG 10         // Tag width for loop predictor

// RL Policy Network Parameters
#define INPUT_SIZE 12   // Increased to include Loop Predictor features
#define HIDDEN_SIZE1 32 // Larger hidden layer for better capacity
#define HIDDEN_SIZE2 16
#define OUTPUT_SIZE 1           // Binary output (flip or no-flip)
#define WEIGHT_PRECISION 8      // 8-bit weights for storage efficiency
#define LEARNING_RATE 0.000001  // Initial learning rate (slightly reduced)
#define WEIGHT_DECAY 0.001f    // L2 regularization factor
#define REPLAY_BUFFER_SIZE 3000 // Buffer size
#define BATCH_SIZE 64           // Batch size for updates

// Global Arrays
bool NOSKIP[NHIST + 1];   // Controls TAGE table associativity
int SizeTable[NHIST + 1]; // Sizes of TAGE tables

// Bimodal Table Entry
class bentry
{
public:
    int8_t hyst; // 1-bit hysteresis
    int8_t pred; // 1-bit prediction

    bentry() : pred(0), hyst(1) {} // Initialize to not taken with hysteresis
};

// Tagged Table Entry
class gentry
{
public:
    int8_t ctr; // 3-bit prediction counter
    uint tag;   // Tag for history matching
    int8_t u;   // 1-bit usefulness counter

    gentry() : ctr(0), tag(0), u(0) {} // Neutral state
};

// Loop Predictor Entry
class lentry
{
public:
    uint16_t NbIter;      // 10-bit iteration count
    uint8_t confid;       // 4-bit confidence
    uint16_t CurrentIter; // 10-bit current iteration
    uint16_t TAG;         // 10-bit tag
    uint8_t age;          // 4-bit age
    bool dir;             // 1-bit direction

    lentry() : NbIter(0), confid(0), CurrentIter(0), TAG(0), age(0), dir(false) {}
};

// Folded History for Index/Tag Compression
class folded_history
{
public:
    unsigned comp; // Compressed history
    int CLENGTH;   // Compressed length
    int OLENGTH;   // Original length
    int OUTPOINT;  // Folding point

    folded_history() {}

    void init(int original_length, int compressed_length)
    {
        comp = 0;
        OLENGTH = original_length;
        CLENGTH = compressed_length;
        OUTPOINT = OLENGTH % CLENGTH;
    }

    void update(std::array<uint8_t, HISTBUFFERLENGTH> &h, int PT)
    {
        comp = (comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)];
        comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
        comp ^= (comp >> CLENGTH);
        comp &= ((1 << CLENGTH) - 1);
    }
};

// Type Aliases for TAGE
using tage_index_t = std::array<folded_history, NHIST + 1>;
using tage_tag_t = std::array<folded_history, NHIST + 1>;

// RL-specific structures
struct RLState
{
    std::array<uint8_t, INPUT_SIZE> features; // 12-bit state
};

struct RLAction
{
    bool flip;  // Action: flip TAGE prediction or not
    float prob; // Probability of flipping
};

struct Experience
{
    RLState state;
    RLAction action;
    float reward;
    Experience(RLState s, RLAction a, float r) : state(s), action(a), reward(r) {}
};

// History Structure (simplified, no SC histories)
struct cbp_hist_t
{
    uint64_t GHIST;                              // Global history
    std::array<uint8_t, HISTBUFFERLENGTH> ghist; // Circular history buffer
    uint64_t phist;                              // Path history
    int ptghist;                                 // Pointer to history buffer
    tage_index_t ch_i;                           // Folded index histories
    std::array<tage_tag_t, 2> ch_t;              // Folded tag histories
};

// Main Predictor Class
class CBP2016_TAGE_SC_L
{
public:
    // TAGE Prediction State
    int GI[NHIST + 1];                // TAGE table indices
    uint GTAG[NHIST + 1];             // TAGE table tags
    int BI;                           // Bimodal index
    bool tage_pred;                   // TAGE prediction
    bool alttaken;                    // Alternate prediction
    bool LongestMatchPred;            // Longest matching prediction
    int HitBank;                      // Index of hit bank
    int AltBank;                      // Index of alternate bank
    bool LowConf;                     // Low confidence flag
    bool HighConf;                    // High confidence flag
    bool AltConf;                     // Confidence on alternate prediction
    int8_t BIM;                       // Bimodal prediction state
    int TICK;                         // Aging counter
    int8_t use_alt_on_na[SIZEUSEALT]; // Alternate prediction confidence table

    // Loop Predictor State
    bool predloop;           // Loop prediction
    int LIB, LI, LHIT, LTAG; // Loop indices and tags
    bool LVALID;             // Loop prediction validity
    int8_t WITHLOOP;         // Loop confidence counter

    // RL Policy Network state
    std::array<std::array<float, HIDDEN_SIZE1>, INPUT_SIZE> w1;   // Input to first hidden
    std::array<std::array<float, HIDDEN_SIZE2>, HIDDEN_SIZE1> w2; // First hidden to second hidden
    std::array<float, HIDDEN_SIZE2> w3;                           // Second hidden to output
    std::array<float, HIDDEN_SIZE1> hidden1;                      // First hidden layer activations
    std::array<float, HIDDEN_SIZE2> hidden2;                      // Second hidden layer activations
    std::vector<Experience> replay_buffer;                        // New member
    RLState pred_state;                                           // Current state
    RLAction pred_action;                                         // Current action
    float current_lr;                                             // Dynamic learning rate

    // History and Tables
    cbp_hist_t active_hist;                                                           // Current history
    std::unordered_map<uint64_t, std::pair<cbp_hist_t, RLState>> pred_time_histories; // Checkpointed history and state
    bentry *btable;                                                                   // Bimodal table
    gentry *gtable[NHIST + 1];                                                        // Tagged tables
    lentry *ltable;                                                                   // Loop table
    int m[NHIST + 1];                                                                 // History lengths
    int TB[NHIST + 1];                                                                // Tag widths
    int logg[NHIST + 1];                                                              // Log sizes
    uint64_t Seed;                                                                    // Random seed

    CBP2016_TAGE_SC_L(void)
    {
        btable = nullptr; // Initialize pointers
        ltable = nullptr;
        for (int i = 1; i <= NHIST; i++)
            gtable[i] = nullptr;
        init_histories(active_hist);
        init_rl_network();
        current_lr = LEARNING_RATE;
        replay_buffer.reserve(REPLAY_BUFFER_SIZE);
#ifdef PRINTSIZE
        predictorsize();
#endif
    }

    // Setup and Teardown (placeholders)
    void setup() {}
    void terminate() {}

    // Unique Instruction ID
    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const
    {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    // Initialize Histories
    void init_histories(cbp_hist_t &current_hist)
    {
        // Set geometric history lengths
        m[1] = MINHIST;
        m[NHIST / 2] = MAXHIST;
        for (int i = 2; i <= NHIST / 2; i++)
        {
            m[i] = (int)(((double)MINHIST * pow((double)MAXHIST / (double)MINHIST, (double)(i - 1) / (double)((NHIST / 2) - 1))) + 0.5);
        }
        for (int i = NHIST; i > 1; i--)
        {
            m[i] = m[(i + 1) / 2];
        }

        // Configure TAGE tables
        for (int i = 1; i <= NHIST; i++)
        {
            TB[i] = TBITS + 4 * (i >= BORN); // 8 bits for low, 12 for high
            logg[i] = LOGG;
            NOSKIP[i] = ((i - 1) & 1) || ((i >= BORNINFASSOC) & (i < BORNSUPASSOC));
        }
        NOSKIP[4] = 0;
        NOSKIP[NHIST - 2] = 0;
        NOSKIP[8] = 0;
        NOSKIP[NHIST - 6] = 0;

        // Allocate tables
#ifdef LOOPPREDICTOR
        ltable = new lentry[1 << LOGL]();
#endif
        gtable[1] = new gentry[NBANKLOW * (1 << LOGG)]();
        SizeTable[1] = NBANKLOW * (1 << LOGG);
        gtable[BORN] = new gentry[NBANKHIGH * (1 << LOGG)]();
        SizeTable[BORN] = NBANKHIGH * (1 << LOGG);
        for (int i = BORN + 1; i <= NHIST; i++)
            gtable[i] = gtable[BORN];
        for (int i = 2; i <= BORN - 1; i++)
            gtable[i] = gtable[1];
        btable = new bentry[1 << LOGB];

        // Initialize folded histories
        for (int i = 1; i <= NHIST; i++)
        {
            current_hist.ch_i[i].init(m[i], logg[i]);
            current_hist.ch_t[0][i].init(current_hist.ch_i[i].OLENGTH, TB[i]);
            current_hist.ch_t[1][i].init(current_hist.ch_i[i].OLENGTH, TB[i] - 1);
        }

        // Initialize state
        LVALID = false;
        WITHLOOP = -1;
        Seed = 0;
        TICK = 0;
        current_hist.phist = 0;
        for (int i = 0; i < HISTBUFFERLENGTH; i++)
            current_hist.ghist[i] = 0;
        current_hist.ptghist = 0;

        // Initialize alternate prediction table
        for (int i = 0; i < SIZEUSEALT; i++)
            use_alt_on_na[i] = 0;
        for (int i = 0; i < (1 << LOGB); i++)
        {
            btable[i].pred = 0;
            btable[i].hyst = 1;
        }
    }

    /** Initialize RL network weights with small random values */
    // Update init_rl_network
    void init_rl_network()
    {
        for (auto &row : w1)
            for (auto &weight : row)
                weight = (float)(rand() % 256 - 128) / 128.0f; // [-1, 1] range
        for (auto &row : w2)
            for (auto &weight : row)
                weight = (float)(rand() % 256 - 128) / 128.0f;
        for (auto &weight : w3)
            weight = (float)(rand() % 256 - 128) / 128.0f;
    }

    // Bimodal Index
    int bindex(UINT64 PC) const
    {
        return ((PC ^ (PC >> LOGB)) & ((1 << LOGB) - 1));
    }

    // Path History Mixing Function
    int F(uint64_t A, int size, int bank) const
    {
        int A1 = (A & ((1 << logg[bank]) - 1));
        int A2 = (A >> logg[bank]);
        if (bank < logg[bank])
            A2 = ((A2 << bank) & ((1 << logg[bank]) - 1)) + (A2 >> (logg[bank] - bank));
        A = A1 ^ A2;
        if (bank < logg[bank])
            A = ((A << bank) & ((1 << logg[bank]) - 1)) + (A >> (logg[bank] - bank));
        return A;
    }

    // TAGE Index
    int gindex(unsigned int PC, int bank, uint64_t hist, const tage_index_t &ch_i) const
    {
        int index = PC ^ (PC >> (abs(logg[bank] - bank) + 1)) ^ ch_i[bank].comp ^ F(hist, (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank], bank);
        return (index & ((1 << logg[bank]) - 1));
    }

    // TAGE Tag
    uint16_t gtag(unsigned int PC, int bank, const tage_tag_t &tag_0_array, const tage_tag_t &tag_1_array) const
    {
        int tag = PC ^ tag_0_array[bank].comp ^ (tag_1_array[bank].comp << 1);
        return (tag & ((1 << TB[bank]) - 1));
    }

    // Counter Update
    void ctrupdate(int8_t &ctr, bool taken, int nbits)
    {
        if (taken)
        {
            if (ctr < ((1 << (nbits - 1)) - 1))
                ctr++;
        }
        else
        {
            if (ctr > -(1 << (nbits - 1)))
                ctr--;
        }
    }

    // Bimodal Prediction
    bool getbim()
    {
        BIM = (btable[BI].pred << 1) + (btable[BI >> HYSTSHIFT].hyst);
        HighConf = (BIM == 0) || (BIM == 3);
        LowConf = !HighConf;
        AltConf = HighConf;
        return (btable[BI].pred > 0);
    }

    // Bimodal Update
    void baseupdate(bool Taken)
    {
        int inter = BIM;
        if (Taken)
        {
            if (inter < 3)
                inter++;
        }
        else if (inter > 0)
            inter--;
        btable[BI].pred = inter >> 1;
        btable[BI >> HYSTSHIFT].hyst = inter & 1;
    }

    // Pseudo-Random Number Generator
    int MYRANDOM()
    {
        Seed++;
        Seed ^= active_hist.phist;
        Seed = (Seed >> 21) + (Seed << 11);
        Seed ^= (int64_t)active_hist.ptghist;
        Seed = (Seed >> 10) + (Seed << 22);
        return (Seed & 0xFFFFFFFF);
    }

    // TAGE Prediction
    void Tagepred(UINT64 PC, const cbp_hist_t &hist_to_use)
    {
        HitBank = 0;
        AltBank = 0;
        for (int i = 1; i <= NHIST; i += 2)
        {
            GI[i] = gindex(PC, i, hist_to_use.phist, hist_to_use.ch_i);
            GTAG[i] = gtag(PC, i, hist_to_use.ch_t[0], hist_to_use.ch_t[1]);
            GTAG[i + 1] = GTAG[i];
            GI[i + 1] = GI[i] ^ (GTAG[i] & ((1 << LOGG) - 1));
        }

        int T = (PC ^ (hist_to_use.phist & ((1ULL << m[BORN]) - 1))) % NBANKHIGH;
        for (int i = BORN; i <= NHIST; i++)
            if (NOSKIP[i])
            {
                GI[i] += (T << LOGG);
                T++;
                T = T % NBANKHIGH;
            }
        T = (PC ^ (hist_to_use.phist & ((1 << m[1]) - 1))) % NBANKLOW;
        for (int i = 1; i <= BORN - 1; i++)
            if (NOSKIP[i])
            {
                GI[i] += (T << LOGG);
                T++;
                T = T % NBANKLOW;
            }

        BI = bindex(PC);
        alttaken = getbim();
        tage_pred = alttaken;
        LongestMatchPred = alttaken;

        for (int i = NHIST; i > 0; i--)
            if (NOSKIP[i])
                if (gtable[i][GI[i]].tag == GTAG[i])
                {
                    HitBank = i;
                    LongestMatchPred = (gtable[HitBank][GI[HitBank]].ctr >= 0);
                    break;
                }

        for (int i = HitBank - 1; i > 0; i--)
            if (NOSKIP[i])
                if (gtable[i][GI[i]].tag == GTAG[i])
                {
                    AltBank = i;
                    break;
                }

        if (HitBank > 0)
        {
            if (AltBank > 0)
            {
                alttaken = (gtable[AltBank][GI[AltBank]].ctr >= 0);
                AltConf = (abs(2 * gtable[AltBank][GI[AltBank]].ctr + 1) > 1);
            }
            else
            {
                alttaken = getbim();
            }
            bool Huse_alt_on_na = (use_alt_on_na[(((HitBank - 1) / 8) << 1) + AltConf] >= 0);
            if (!Huse_alt_on_na || (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) > 1))
                tage_pred = LongestMatchPred;
            else
                tage_pred = alttaken;
            HighConf = (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) >= (1 << CWIDTH) - 1);
            LowConf = (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1);
        }
    }

    /** Compute state with enriched features */
    RLState compute_state(UINT64 PC, const cbp_hist_t &hist)
    {
        RLState state;
        uint64_t hash = PC ^ (hist.GHIST & 0xFFF) ^ (hist.phist & 0xFFF);
        state.features[0] = tage_pred ? 1 : 0; // TAGE prediction
        state.features[1] = HighConf ? 1 : 0;  // High confidence
        state.features[2] = LowConf ? 1 : 0;   // Low confidence
        state.features[3] = HitBank & 0xFF;    // TAGE hit bank (new feature)
#ifdef LOOPPREDICTOR
        state.features[4] = LVALID ? 1 : 0;                // Loop validity
        state.features[5] = (WITHLOOP >= 0) ? 1 : 0;       // Loop confidence
        state.features[6] = ltable[LI].CurrentIter & 0xFF; // Loop iteration count (new feature)
#else
        state.features[4] = 0;
        state.features[5] = 0;
        state.features[6] = 0;
#endif
        for (int i = 7; i < INPUT_SIZE; i++)
        {
            state.features[i] = (hash >> (i - 7)) & 1; // Additional hash bits
        }
        return state;
    }

    /** Predict action using the RL policy network */
    RLAction rl_predict(const RLState &state)
    {
        // Input -> Hidden1 (ReLU)
        for (int j = 0; j < HIDDEN_SIZE1; j++)
        {
            float sum = 0.0f;
            for (int i = 0; i < INPUT_SIZE; i++)
                sum += state.features[i] * w1[i][j];
            hidden1[j] = (sum > 0) ? sum : 0.01*sum;
        }

        // Hidden1 -> Hidden2 (ReLU)
        for (int j = 0; j < HIDDEN_SIZE2; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < HIDDEN_SIZE1; k++)
                sum += hidden1[k] * w2[k][j];
            hidden2[j] = (sum > 0) ? sum : 0.01*sum;
        }

        // Hidden2 -> Output (sigmoid)
        float logit = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE2; j++)
            logit += hidden2[j] * w3[j];

        // Clip logit to prevent overflow
        
        logit = std::max(-40.0f, std::min(40.0f, logit));
        // Use numerically stable sigmoid calculation
        float prob = 1.0f / (1.0f + expf(-logit));

        // Use better random number generation
        bool flip = (float)(rand() % 1000) / 1000.0f < prob;
        //std::cout<<prob<<","<<logit<<","<<flip<<std::endl;
        
        return {flip, prob};
    }

    // Main Prediction Function
    bool predict(uint64_t seq_no, uint8_t piece, UINT64 PC)
    {
        pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), std::make_pair(active_hist, compute_state(PC, active_hist)));
        return predict_using_given_hist(seq_no, piece, PC, active_hist, true);
    }

    // Prediction with Given History
    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, UINT64 PC, const cbp_hist_t &hist_to_use, const bool pred_time_predict)
    {
        Tagepred(PC, hist_to_use);
        bool pred_taken = tage_pred;

#ifdef LOOPPREDICTOR
        predloop = getloop(PC, hist_to_use);
        pred_taken = ((WITHLOOP >= 0) && LVALID) ? predloop : pred_taken;
#endif

        if (pred_time_predict)
        {
            pred_state = compute_state(PC, hist_to_use);
            pred_action = rl_predict(pred_state);
        }
        pred_taken = pred_action.flip ? !pred_taken : pred_taken;

        return pred_taken;
    }

    // History Update (for branches)
    void history_update(uint64_t seq_no, uint8_t piece, UINT64 PC, int brtype, bool pred_taken, bool taken, UINT64 nextPC)
    {
        HistoryUpdate(PC, brtype, taken, nextPC);
    }

    // History Update (for non-branches)
    void TrackOtherInst(UINT64 PC, int brtype, bool pred_dir, bool taken, UINT64 nextPC)
    {
        HistoryUpdate(PC, brtype, taken, nextPC);
    }

    // Common History Update
    void HistoryUpdate(UINT64 PC, int brtype, bool taken, UINT64 nextPC)
    {
        auto &X = active_hist.phist;
        auto &Y = active_hist.ptghist;
        auto &H = active_hist.ch_i;
        auto &G = active_hist.ch_t[0];
        auto &J = active_hist.ch_t[1];
        int maxt = (brtype & 1) ? 2 : ((brtype & 2) ? 3 : 2);

        if (brtype & 1)
        {
            active_hist.GHIST = (active_hist.GHIST << 1) + (taken & (nextPC < PC));
        }

        int T = ((PC ^ (PC >> 2))) ^ taken;
        int PATH = PC ^ (PC >> 2) ^ (PC >> 4);
        if ((brtype == 3) & taken)
        {
            T = (T ^ (nextPC >> 2));
            PATH = PATH ^ (nextPC >> 2) ^ (nextPC >> 4);
        }

        for (int t = 0; t < maxt; t++)
        {
            bool DIR = (T & 1);
            T >>= 1;
            int PATHBIT = (PATH & 127);
            PATH >>= 1;
            Y--;
            active_hist.ghist[Y & (HISTBUFFERLENGTH - 1)] = DIR;
            X = (X << 1) ^ PATHBIT;
            for (int i = 1; i <= NHIST; i++)
            {
                H[i].update(active_hist.ghist, Y);
                G[i].update(active_hist.ghist, Y);
                J[i].update(active_hist.ghist, Y);
            }
        }
        X = (X & ((1 << PHISTWIDTH) - 1));
    }

    // Main Update Function
    void update(uint64_t seq_no, uint8_t piece, UINT64 PC, bool resolveDir, bool predDir, UINT64 nextPC)
    {

        const auto pred_hist_key = get_unique_inst_id(seq_no, piece); // Assumed function
        const auto &entry = pred_time_histories.at(pred_hist_key);
        const auto &pred_time_history = entry.first;
        const auto &pred_time_state = entry.second;

        const bool pred_taken = predict_using_given_hist(seq_no, piece, PC, pred_time_history, false);
    float reward;
        if (pred_taken == resolveDir)
        {
            reward = HighConf ? 1.0f : 0.5f; // Reward based on confidence
        }
        else
        {
            reward = HighConf ? -1.0f : -0.5f; // Penalty based on confidence
        }

        // Store experience
        replay_buffer.push_back(Experience(pred_time_state, pred_action, reward));
        if (replay_buffer.size() > REPLAY_BUFFER_SIZE)
            replay_buffer.erase(replay_buffer.begin());
        pred_time_histories.erase(pred_hist_key);

        /// Perform batch update if enough experiences
         if (replay_buffer.size() >= BATCH_SIZE) {
            update_rl_batch();
         }

        update_rl(PC, resolveDir, pred_taken, pred_time_state);

#ifdef LOOPPREDICTOR
        if (LVALID && pred_taken != predloop)
        {
            ctrupdate(WITHLOOP, (predloop == resolveDir), 7);
        }
        loopupdate(PC, resolveDir, (pred_taken != resolveDir), pred_time_history);
#endif

        // TAGE Update
        bool ALLOC = ((tage_pred != resolveDir) & (HitBank < NHIST));
        if (HitBank > 0)
        {
            bool PseudoNewAlloc = (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) <= 1);
            if (PseudoNewAlloc)
            {
                if (LongestMatchPred == resolveDir)
                    ALLOC = false;
                if (LongestMatchPred != alttaken)
                {
                    ctrupdate(use_alt_on_na[(((HitBank - 1) / 8) << 1) + AltConf], (alttaken == resolveDir), ALTWIDTH);
                }
            }
        }
        if (pred_taken == resolveDir && (MYRANDOM() & 31) != 0)
            ALLOC = false;

        if (ALLOC)
        {
            int T = NNN;
            int A = (MYRANDOM() & 127) < 32 ? 2 : 1;
            int Penalty = 0;
            int NA = 0;
            int DEP = ((((HitBank - 1 + 2 * A) & 0xffe)) ^ (MYRANDOM() & 1));
            for (int I = DEP; I < NHIST; I += 2)
            {
                int i = I + 1;
                bool Done = false;
                if (NOSKIP[i] && gtable[i][GI[i]].u == 0)
                {
                    if (abs(2 * gtable[i][GI[i]].ctr + 1) <= 3)
                    {
                        gtable[i][GI[i]].tag = GTAG[i];
                        gtable[i][GI[i]].ctr = resolveDir ? 0 : -1;
                        NA++;
                        if (T <= 0)
                            break;
                        I += 2;
                        Done = true;
                        T -= 1;
                    }
                    else
                    {
                        if (gtable[i][GI[i]].ctr > 0)
                            gtable[i][GI[i]].ctr--;
                        else
                            gtable[i][GI[i]].ctr++;
                    }
                }
                else
                {
                    Penalty++;
                }
                if (!Done)
                {
                    i = (I ^ 1) + 1;
                    if (NOSKIP[i] && gtable[i][GI[i]].u == 0)
                    {
                        if (abs(2 * gtable[i][GI[i]].ctr + 1) <= 3)
                        {
                            gtable[i][GI[i]].tag = GTAG[i];
                            gtable[i][GI[i]].ctr = resolveDir ? 0 : -1;
                            NA++;
                            if (T <= 0)
                                break;
                            I += 2;
                            T -= 1;
                        }
                        else
                        {
                            if (gtable[i][GI[i]].ctr > 0)
                                gtable[i][GI[i]].ctr--;
                            else
                                gtable[i][GI[i]].ctr++;
                        }
                    }
                    else
                    {
                        Penalty++;
                    }
                }
            }
            TICK += (Penalty - 2 * NA);
            if (TICK < 0)
                TICK = 0;
            if (TICK >= BORNTICK)
            {
                for (int i = 1; i <= BORN; i += BORN - 1)
                    for (int j = 0; j < SizeTable[i]; j++)
                        gtable[i][j].u >>= 1;
                TICK = 0;
            }
        }

        if (HitBank > 0)
        {
            if (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1)
            {
                if (LongestMatchPred != resolveDir)
                {
                    if (AltBank > 0)
                        ctrupdate(gtable[AltBank][GI[AltBank]].ctr, resolveDir, CWIDTH);
                    else
                        baseupdate(resolveDir);
                }
            }
            ctrupdate(gtable[HitBank][GI[HitBank]].ctr, resolveDir, CWIDTH);
            if (abs(2 * gtable[HitBank][GI[HitBank]].ctr + 1) == 1)
                gtable[HitBank][GI[HitBank]].u = 0;
            if (alttaken == resolveDir && AltBank > 0 && abs(2 * gtable[AltBank][GI[AltBank]].ctr + 1) == 7 && gtable[HitBank][GI[HitBank]].u == 1)
            {
                if (LongestMatchPred == resolveDir)
                    gtable[HitBank][GI[HitBank]].u = 0;
            }
        }
        else
        {
            baseupdate(resolveDir);
        }
        if (LongestMatchPred != alttaken && LongestMatchPred == resolveDir)
        {
            if (gtable[HitBank][GI[HitBank]].u < (1 << UWIDTH) - 1)
                gtable[HitBank][GI[HitBank]].u++;
        }

        // Learning rate decay every 500 updates
        static int update_count = 0;
        if (++update_count % 1000 == 0)
        {
            current_lr *= 0.99f; // 5% decay (was 0.99f every 1000 updates)
        }
    }

    // RL Update
    void update_rl(UINT64 PC, bool resolveDir, bool pred_taken, const RLState &state)
    {
        //  float reward = (pred_taken == resolveDir) ? 1.0f : -1.0f;  // +1 for correct, -1 for incorrect
        float reward = 0.0f; // +1 for correct, -1 for incorrect
        if (pred_action.prob > 0.4f)
        {
            reward = (pred_taken == resolveDir) ? 5.0f: -5.0f; // Double reward for high confidence
        }
        else if (pred_action.prob > 0.1f)
        {
            reward = (pred_taken == resolveDir) ? 0.75f : -0.75f; // Double penalty for low confidence
        }
        else
        {
            reward = (pred_taken == resolveDir) ? 0.5f : -2.5f; // Neutral reward for low confidence
        }

        

        float grad_log_pi = pred_action.flip ? (1.0f - pred_action.prob) : -pred_action.prob;

        // Update weights with policy gradient and L2 regularization
        for (int j = 0; j < HIDDEN_SIZE2; j++)
        {
            float grad_w3 = hidden2[j] * grad_log_pi * reward - WEIGHT_DECAY * w3[j];
            w3[j] += current_lr * grad_w3;
            for (int i = 0; i < HIDDEN_SIZE1; i++)
            {
                float grad_w2 = hidden1[i] * (hidden2[j] > 0 ? 1.0f : 0.0f) * w3[j] * grad_log_pi * reward - WEIGHT_DECAY * w2[i][j];
                w2[i][j] += current_lr * grad_w2;
            }
        }
        for (int j = 0; j < HIDDEN_SIZE1; j++)
        {
            for (int i = 0; i < INPUT_SIZE; i++)
            {
                float grad_w1 = state.features[i] * (hidden1[j] > 0 ? 1.0f : 0.0f) * w2[j][0] * grad_log_pi * reward - WEIGHT_DECAY * w1[i][j];
                w1[i][j] += current_lr * grad_w1;
            }
        }

        // Clip weights to 8-bit range
        for (int i = 0; i < INPUT_SIZE; i++)
            for (int j = 0; j < HIDDEN_SIZE1; j++)
                w1[i][j] = std::max(-128.0f, std::min(127.0f, w1[i][j]));
        for (int i = 0; i < HIDDEN_SIZE1; i++)
            for (int j = 0; j < HIDDEN_SIZE2; j++)
                w2[i][j] = std::max(-128.0f, std::min(127.0f, w2[i][j]));
        for (int j = 0; j < HIDDEN_SIZE2; j++)
            w3[j] = std::max(-128.0f, std::min(127.0f, w3[j]));
    }

#ifdef LOOPPREDICTOR
    // Loop Predictor Functions
    bool getloop(UINT64 PC, const cbp_hist_t &hist_to_use)
    {
        LI = (PC ^ (PC >> 2)) & ((1 << LOGL) - 1);
        LIB = LI;
        LHIT = -1;
        int maxt = 2; // Fixed value instead of depending on nextPC

        for (int t = 0; t < maxt; t++)
        {
            bool DIR = ((PC ^ (PC >> 2)) >> t) & 1;
            if (ltable[LIB].dir == DIR)
            {
                if (ltable[LIB].CurrentIter + 1 == ltable[LIB].NbIter && ltable[LIB].confid >= 8)
                {
                    LHIT = t;
                    LTAG = ltable[LIB].TAG;
                    break;
                }
            }
            LIB = (LIB + 1) & ((1 << LOGL) - 1);
        }

        LVALID = (LHIT >= 0);
        return (LVALID ? ltable[LI].dir : false);
    }

    void loopupdate(UINT64 PC, bool resolveDir, bool pred_wrong, const cbp_hist_t &hist_to_use)
    {
        if (LVALID)
        {
            if (resolveDir != predloop)
            {
                if (ltable[LI].CurrentIter + 1 == ltable[LI].NbIter)
                {
                    if (ltable[LI].confid < 15)
                        ltable[LI].confid++;
                }
                else
                {
                    ltable[LI].NbIter = 0;
                    ltable[LI].age = 0;
                    ltable[LI].confid = 0;
                    ltable[LI].CurrentIter = 0;
                }
            }
            else
            {
                if (ltable[LI].CurrentIter + 1 != ltable[LI].NbIter)
                {
                    if (ltable[LI].age < 15)
                        ltable[LI].age++;
                }
            }
            ltable[LI].CurrentIter = (ltable[LI].CurrentIter + 1) & ((1 << WIDTHNBITERLOOP) - 1);
        }

        if (pred_wrong && !LVALID)
        {
            int index = (PC ^ (PC >> 2)) & ((1 << LOGL) - 1);
            if (ltable[index].age == 0)
            {
                ltable[index].dir = resolveDir;
                ltable[index].TAG = (PC >> LOGL) & ((1 << LOOPTAG) - 1);
                ltable[index].NbIter = 0;
                ltable[index].age = 0;
                ltable[index].confid = 0;
                ltable[index].CurrentIter = 0;
            }
            else
            {
                ltable[index].age--;
            }
        }
    }
#endif

    // Storage Size Calculation
    int predictorsize()
    {
        int STORAGESIZE = 0;

        // TAGE Storage
        STORAGESIZE += NBANKHIGH * (1 << logg[BORN]) * (CWIDTH + UWIDTH + TB[BORN]); // High history tables
        STORAGESIZE += NBANKLOW * (1 << logg[1]) * (CWIDTH + UWIDTH + TB[1]);        // Low history tables
        STORAGESIZE += SIZEUSEALT * ALTWIDTH;                                        // Alternate prediction table
        STORAGESIZE += (1 << LOGB) + (1 << (LOGB - HYSTSHIFT));                      // Bimodal table
        STORAGESIZE += m[NHIST];                                                     // Max history length
        STORAGESIZE += PHISTWIDTH;                                                   // Path history
        STORAGESIZE += 10;                                                           // TICK counter

#ifdef LOOPPREDICTOR
        // Loop Predictor Storage
        STORAGESIZE += (1 << LOGL) * (2 * WIDTHNBITERLOOP + LOOPTAG + 4 + 4 + 1); // 32 * 39 bits
#endif

        // Updated RL storage
        STORAGESIZE += INPUT_SIZE * HIDDEN_SIZE1 * WEIGHT_PRECISION;   // w1: 16 * 64 * 8 bits
        STORAGESIZE += HIDDEN_SIZE1 * HIDDEN_SIZE2 * WEIGHT_PRECISION; // w2: 64 * 32 * 8 bits
        STORAGESIZE += HIDDEN_SIZE2 * WEIGHT_PRECISION;                // w3: 32 * 8 bits
        STORAGESIZE += HIDDEN_SIZE1 * 32;                              // hidden1: 64 * 32-bit floats
        STORAGESIZE += HIDDEN_SIZE2 * 32;                              // hidden2: 32 * 32-bit floats
        STORAGESIZE += REPLAY_BUFFER_SIZE * (INPUT_SIZE * 8 + 1 + 32); // replay buffer: 10000 * (16*8 + 1 + 32) bits

        fprintf(stderr, " (TOTAL %d bits %d Kbits) ", STORAGESIZE, STORAGESIZE / 1024);
        return STORAGESIZE;
    }

private:
    void update_rl_batch()
    {
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            int idx = rand() % replay_buffer.size();
            const auto &exp = replay_buffer[idx];
            RLState state = exp.state;
            RLAction action = exp.action;
            float reward = exp.reward;

            // Forward pass
            for (int j = 0; j < HIDDEN_SIZE1; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < INPUT_SIZE; k++)
                    sum += state.features[k] * w1[k][j];
                hidden1[j] = (sum > 0) ? sum : 0;
            }
            for (int j = 0; j < HIDDEN_SIZE2; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < HIDDEN_SIZE1; k++)
                    sum += hidden1[k] * w2[k][j];
                hidden2[j] = (sum > 0) ? sum : 0;
            }
            float logit = 0.0f;
            for (int j = 0; j < HIDDEN_SIZE2; j++)
                logit += hidden2[j] * w3[j];
            float prob = 1.0f / (1.0f + expf(-logit));
            float grad_log_pi = action.flip ? (1.0f - prob) : -prob;

            // Update weights
            for (int j = 0; j < HIDDEN_SIZE2; j++)
            {
                float grad_w3 = hidden2[j] * grad_log_pi * reward - WEIGHT_DECAY * w3[j];
                w3[j] += current_lr * grad_w3;
                for (int k = 0; k < HIDDEN_SIZE1; k++)
                {
                    float grad_w2 = hidden1[k] * (hidden2[j] > 0 ? 1.0f : 0.0f) * w3[j] * grad_log_pi * reward - WEIGHT_DECAY * w2[k][j];
                    w2[k][j] += current_lr * grad_w2;
                }
            }
            for (int j = 0; j < HIDDEN_SIZE1; j++)
            {
                for (int k = 0; k < INPUT_SIZE; k++)
                {
                    float grad_w1 = state.features[k] * (hidden1[j] > 0 ? 1.0f : 0.0f) * w2[j][0] * grad_log_pi * reward - WEIGHT_DECAY * w1[k][j];
                    w1[k][j] += current_lr * grad_w1;
                }
            }

            // Clip weights
            for (auto &row : w1)
                for (auto &weight : row)
                    weight = std::max(-128.0f, std::min(127.0f, weight));
            for (auto &row : w2)
                for (auto &weight : row)
                    weight = std::max(-128.0f, std::min(127.0f, weight));
            for (auto &weight : w3)
                weight = std::max(-128.0f, std::min(127.0f, weight));
        }
    }
};

// Static Instance
static CBP2016_TAGE_SC_L cbp2016_tage_sc_l;

#endif // _TAGE_PREDICTOR_H_
