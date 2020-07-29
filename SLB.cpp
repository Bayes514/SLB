/* 
 * File:   SLB.cpp
 * Author: Yang Liu
 * 
 * Created on 2020年2月23日, 下午4:52
 */
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "SLB.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

SLB::SLB() :
trainingIsFinished_(false) {
}

SLB::SLB(char*const*& argv, char*const* end) :
trainingIsFinished_(false) {
    name_ = "SLB";

    // defaults
    k_ = 1;
    printm = false;
    // get arguments
    while (argv != end) {
        if (*argv[0] != '-') {
            break;
        } else if (argv[0][1] == 'k') {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else if (streq(argv[0] + 1, "p")) {
            printm = true;
        } else {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

SLB::~SLB(void) {
}

void SLB::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass {
public:

    miCmpClass(std::vector<double> *m) {
        mimi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b) {
        return (*mimi)[a] > (*mimi)[b];
    }

private:
    std::vector<double> *mimi;
};

void SLB::reset(InstanceStream &is) {
    //printf("reset\n");
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1

    // initialise distributions
    parents_3d.resize(noClasses_);
    mi.resize(noCatAtts_); //n*y
    cmi.resize(noCatAtts_); //n*n*y

    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        parents_3d[a].resize(noCatAtts);
    }
    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts; b++) {
            parents_3d[a][b].clear();
        }
    }

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        mi[a].clear();
    }
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        cmi[i].resize(noCatAtts_);
        for (unsigned int j = 0; j < noCatAtts_; j++)
            cmi[i][j].clear();
    }

    /*初始化各数据结构空间*/
    dist_.reset(is); //

    trainingIsFinished_ = false;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void SLB::train(const instance &inst) {
    //printf("train\n");
    dist_.update(inst);
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void SLB::initialisePass() {
    assert(trainingIsFinished_ == false);
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void SLB::finalisePass() {
    assert(trainingIsFinished_ == false);
    //printf("finalisePass\n");


    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        mi[a].assign(noClasses_, 0.0);
    }
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        cmi[i].resize(noCatAtts_);
        for (unsigned int j = 0; j < noCatAtts_; j++)
            cmi[i][j].assign(noClasses_, 0);
    }

    getMutualInformationTC(dist_.xxyCounts.xyCounts, mi);
    getCondMutualInfTC(dist_.xxyCounts, cmi);

    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool SLB::trainingIsFinished() {
    return trainingIsFinished_;
}

void SLB::classify(const instance& inst, std::vector<double> &posteriorDist) {
    if (printm) printf("classify\n");

    std::vector<std::vector<float> > mi_loc; //n*y
    mi_loc.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        mi_loc[a].assign(noClasses_, 0.0);
    }
    getMutualInformationlocloc(dist_.xxyCounts.xyCounts, mi_loc, inst); //instance-based mi

    std::vector<std::vector<std::vector<float> > > cmi_loc; //n*n*y
    cmi_loc.resize(noCatAtts_);
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        cmi_loc[i].resize(noCatAtts_);
        for (unsigned int j = 0; j < noCatAtts_; j++)
            cmi_loc[i][j].assign(noClasses_, 0);
    }
    getCondMutualInflocloc(dist_.xxyCounts, cmi_loc, inst); //instance-based cmi

    //取平均
    std::vector<std::vector<float> > ami; //n*y
    ami.resize(noCatAtts_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        ami[a].assign(noClasses_, 0.0);
    }
    std::vector<std::vector<std::vector<float> > > acmi; //n*n*y
    acmi.resize(noCatAtts_);
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        acmi[i].resize(noCatAtts_);
        for (unsigned int j = 0; j < noCatAtts_; j++)
            acmi[i][j].assign(noClasses_, 0);
    }

    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        for (CategoricalAttribute b = 0; b < noClasses_; b++) {
            ami[a][b] = (mi[a][b] + mi_loc[a][b]) / 2;
        }
    }
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts_; b++) {
            for (CategoricalAttribute c = 0; c < noClasses_; c++) {
                acmi[a][b][c] = (cmi[a][b][c] + cmi_loc[a][b][c]) / 2;
            }
        }
    }


    for (CategoricalAttribute a = 0; a < noClasses_; a++) {
        for (CategoricalAttribute b = 0; b < noCatAtts_; b++) {
            parents_3d[a][b].clear();
        }
    }
    std::vector<CategoricalAttribute> order;
    std::vector<double> ami_temp; //1*n

    // proper KDB assignment of parents
    for (CategoricalAttribute y = 0; y < noClasses_; y++) {
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            order.push_back(a);
        }
        ami_temp.assign(noCatAtts_, 0.0);

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
            ami_temp[a] = ami[a][y];
        }

        miCmpClass cmp(&ami_temp);
        std::sort(order.begin(), order.end(), cmp);

        for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++) {
            parents_3d[y][*it].push_back(order[0]);
            for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++) {
                // make parents into the top k attributes on mi that precede *it in order
                if (parents_3d[y][*it].size() < k_) {
                    // create space for another parent
                    // set it initially to the new parent.
                    // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                    parents_3d[y][*it].push_back(*it2);
                }
                for (unsigned int i = 0; i < parents_3d[y][*it].size(); i++) {
                    if (acmi[*it][*it2][y] > acmi[*it][parents_3d[y][*it][i]][y]) {
                        // move lower value parents down in order
                        for (unsigned int j = parents_3d[y][*it].size() - 1; j > i; j--) {
                            parents_3d[y][*it][j] = parents_3d[y][*it][j - 1];
                        }
                        // insert the new att
                        parents_3d[y][*it][i] = *it2;
                        break;
                    }
                }
            }
        }
        if (printm) {
            printf("y=%d\tparents_after:\n", y);
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++) {
                if (parents_3d[y][*it].size() == 0) {
                    printf("%d\n", *it);
                }
                if (parents_3d[y][*it].size() > 0) {
                    for (int i = 0; i < parents_3d[y][*it].size(); i++)
                        printf("%d\t%d\t%lf\n", parents_3d[y][*it][i], *it, acmi[parents_3d[y][*it][i]][*it][y]);
                }
            }
            printf("\n");
        }
        order.clear();
        ami_temp.clear();
    }


    //算后验
    crosstab<double> posteriorDist_2d = crosstab<double>(noClasses_);
    for (unsigned int y = 0; y < noClasses_; y++) {
        posteriorDist_2d[y].assign(noClasses_, 0);
    }

    for (CatValue c = 0; c < noClasses_; c++) {
        // calculate the class probabilities in parallel
        // P(y)
        for (CatValue y = 0; y < noClasses_; y++) {
            posteriorDist_2d[c][y] = dist_.xxyCounts.xyCounts.p(y)* (std::numeric_limits<double>::max() / 2.0);
        }

        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            for (CatValue y = 0; y < noClasses_; y++) {
                if (parents_3d[c][x1].size() == 0) {
                    posteriorDist_2d[c][y] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y); // p(a=v|Y=y) using M-estimate
                } else if (parents_3d[c][x1].size() == 1) {
                    const InstanceCount totalCount1 = dist_.xxyCounts.xyCounts.getCount(parents_3d[c][x1][0], inst.getCatVal(parents_3d[c][x1][0]));
                    if (totalCount1 == 0) {
                        posteriorDist_2d[c][y] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                    } else {
                        posteriorDist_2d[c][y] *= dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_3d[c][x1][0], inst.getCatVal(parents_3d[c][x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                    }
                } else if (parents_3d[c][x1].size() == 2) {
                    const InstanceCount totalCount1 = dist_.xxyCounts.getCount(parents_3d[c][x1][0], inst.getCatVal(parents_3d[c][x1][0]), parents_3d[c][x1][1], inst.getCatVal(parents_3d[c][x1][1]));
                    if (totalCount1 == 0) {
                        const InstanceCount totalCount2 = dist_.xxyCounts.xyCounts.getCount(parents_3d[c][x1][0], inst.getCatVal(parents_3d[c][x1][0]));
                        if (totalCount2 == 0) {
                            posteriorDist_2d[c][y] *= dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);
                        } else {
                            posteriorDist_2d[c][y] *= dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_3d[c][x1][0], inst.getCatVal(parents_3d[c][x1][0]), y);
                        }
                    } else {
                        posteriorDist_2d[c][y] *= dist_.p(x1, inst.getCatVal(x1), parents_3d[c][x1][0], inst.getCatVal(parents_3d[c][x1][0]), parents_3d[c][x1][1], inst.getCatVal(parents_3d[c][x1][1]), y);
                    }
                }
            }
        }
    }
//        for (CatValue c = 0; c < noClasses_; c++) {
//            for (CatValue y = 0; y < noClasses_; y++) {
//                if(c == y)
//                    posteriorDist[c] = posteriorDist_2d[c][y];
//            }
//        }

    for (CatValue c = 0; c < noClasses_; c++) {
        for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[y] += posteriorDist_2d[c][y];
        }
    }

    // normalise the results
    normalise(posteriorDist);
}




