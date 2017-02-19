var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index');
  console.log("============")

});

router.post('/', function(req, res, next) {
  // res.render('index');
  res.redirect("http://www.google.com")
 
});

module.exports = router;
