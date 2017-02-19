const Stock = require('./stocksModel')
const PythonShell = require('python-shell');

module.exports = {
  getStock
}

function getStock(req,res){
    var options = {
      mode: 'text',
      pythonPath: '/Users/conway/anaconda3/bin/python',
      pythonOptions: ['-u'],
      scriptPath: '/Users/conway/Documents/pecuniaexmachina/server/stocks',
      args: ['value1', 'value2', 'value3']
    };

    PythonShell.run('script.py', options, function (err, results) {
    // if (err) throw err;
    // results is an array consisting of messages collected during execution
    console.log('results: %j', results);
  })
  .then(() => { res.status(200).end(); })
  .catch((err) => { console.log(err); });
}
