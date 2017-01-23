#include<iostream>
#include<cmath>
#include<vector>
#include<cstdlib>

using namespace std;


double eta = 0.15;
double alpha = 0.5;
class Neuron;
typedef vector<Neuron> Layer;
typedef vector<Layer> Network;

class Neuron{
public:
  double outputVal;
  vector<double> weights;
  vector<double> deltaWeights;
  double gradient;

};



void createNetwork(Network &net, vector<unsigned> &topology){
  unsigned layerCount = topology.size();
  for(unsigned n=0; n<layerCount; ++n){
    net.push_back(Layer());
    for(unsigned m=0; m<=topology[n]; ++m){
      net.back().push_back(Neuron());
      net.back().back().gradient = 0.0;
      net.back().back().outputVal =0.0;
      cout<<"Neuron created. Layer: "<<n<<" Neuron: "<<m<<endl;
      unsigned buffer = n == layerCount-1 ? 0 : topology[n+1];
      for(unsigned q=0; q<buffer; ++q){
        net.back().back().weights.push_back( rand()/double(RAND_MAX));
        net.back().back().deltaWeights.push_back(0.0);
        cout<<net.back().back().weights[q]<<endl<<endl;
    }
      net.back().back().outputVal = 1.0;
    }
  }
}

void feedForward(Network &net, const vector<double> &inputVals){
  double sum = 0.0;
  cout<<endl<<endl<<"Feeding Forward!"<<endl;
  unsigned inputValCount = inputVals.size();
  for(unsigned n=0; n<inputValCount; ++n){
    net[0][n].outputVal = inputVals[n];
  }

for(unsigned layer=0; layer<net.size()-1; ++layer){
  for(unsigned neuronNum=0; neuronNum<net[layer+1].size()-1;++neuronNum){//-1 for bias
    for(unsigned q=0;q<net[layer].size();++q){
      sum+=net[layer][q].weights[neuronNum] * net[layer][q].outputVal;
      if(q==net[layer].size()-1){
      net[layer+1][neuronNum].outputVal = tanh(sum);
      sum= 0.0;
    }
    }
  }
}

}


void backPropagation(Network &net, vector<double> &targetVals, vector<double> &outputVals){
  vector<double> delta;
  Layer &outputLayer = net[net.size()-1];
  //gradient calculation for outputLayer

for(unsigned c=0; c<targetVals.size(); ++c){
    delta.push_back(0.0);
  delta[c] = targetVals[c] - outputVals[c];
  outputLayer[c].gradient = delta[c] * (1.0 - outputLayer[c].outputVal * outputLayer[c].outputVal); //derivative of tanh function
  cout<<endl<<"OutputNeuron: "<<c<<" Gradient: "<<outputLayer[c].gradient;
}
//calculating for hidden layer's gradient
cout<<endl<<"Output layer's gradient calculation finished!"<<endl;
double sum = 0.0;

for(unsigned x=(net.size()-1);x>0;--x){
  for(unsigned k=0;k<net[x-1].size(); ++k){
for(unsigned y=0; y<net[x].size()-1;++y){
sum+=net[x-1][k].weights[y]*net[x][y].gradient;
    if(y==net[x].size()-1){
      net[x-1][k].gradient = sum * (1.0 - net[x-1][k].outputVal * net[x-1][k].outputVal ) ;
      sum =0.0;
      cout<<endl<<"Layer: "<<x-1<<" Neuron: "<<k<<" Gradient: "<<net[x-1][k].gradient;
    }
  }}}

cout<<endl<<"Gradient of hidden layer calculated!"<<endl;
//updating weights.

cout<<endl<<endl<<"Updated Weights List: "<<endl;
for(unsigned x=0; x<net.size()-1; ++x){
  for(unsigned y=0;y<net[x].size(); ++y){
    for(unsigned b=0; b<net[x+1].size()-1;++b){//for bias neuron
  double oldDeltaWeight= net[x][y].deltaWeights[b];

    double newDeltaWeight= eta
    * net[x][y].outputVal
    * net[x][y].gradient
    +alpha*oldDeltaWeight;

    net[x][y].deltaWeights[b] = newDeltaWeight;
    net[x][y].weights[b]+= newDeltaWeight;

    cout<<endl<<"Layer: "<<x<<" Neuron: "<<y<<" Connection: "<<b<<" Weight: "<<net[x][y].weights[b];

  }}
}
}


void getResults(Network &net,vector<double> &results){
  results.clear();
  for(unsigned n =0; n<net.back().size()-1; ++n){
    results.push_back(net.back()[n].outputVal);
  }
}

int main(void){

Network net;
vector<unsigned> topology;
topology.push_back(2);
topology.push_back(3);
topology.push_back(1);

createNetwork(net, topology);
vector<double> inputVals;
inputVals.push_back(0.0);
inputVals.push_back(1.0);

feedForward(net, inputVals);
vector<double> results;
getResults(net, results);
for(unsigned b=0; b<results.size(); ++b){
  cout<<results[b]<<endl;
}
vector<double>targetVals;
targetVals.push_back(0);

for(unsigned i=0;i<1024;i++){
  feedForward(net, inputVals);
  getResults(net, results);
  for(unsigned b=0; b<results.size(); ++b){
    cout<<results[b]<<endl;
  }
 backPropagation(net, targetVals, results);


}
return 0;
}
