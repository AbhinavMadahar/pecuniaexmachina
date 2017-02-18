const Sequelize = require('sequelize');
const POSTGRESS_URL = process.env.POSTGRESS_URL || "postgres://postgres@localhost/stocks"

const sequelize = new Sequelize(POSTGRESS_URL);
sequelize.authenticate()
  .then(() => console.log('database is connected'))
  .catch(err => console.error(err));

module.exports = sequelize;
