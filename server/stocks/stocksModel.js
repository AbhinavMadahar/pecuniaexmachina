const Sequelize = require('sequelize');
const sequelize = require('../config/sequelize');

const Stock = sequelize.define('stock', {
  date: { type: Sequelize.DATE },
  open: { type: Sequelize.DOUBLE },
  close: { type: Sequelize.DOUBLE },
  high: { type: Sequelize.DOUBLE },
  low: { type: Sequelize.DOUBLE }
});

Stock.sync();

module.exports = Stock;
