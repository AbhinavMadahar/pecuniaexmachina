const Stock = require('./stocksModel')

module.exports = {
  addStock
}

function addStock(req,res){
  Stock.create({
    date: req.body.date,
    open: req.body.open,
    close: req.body.close,
    high: req.body.high,
    low: req.body.low
  })
  .then(() => { res.status(200).end(); })
  .catch((err) => { console.log(err); });
}
