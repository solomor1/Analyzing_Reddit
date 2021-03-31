
function autocomplete_subs(inp, arr) {
// https://www.w3schools.com/howto/howto_js_autocomplete.asp
    var currentState;
    // do something when user writes
    inp.addEventListener("input", function(e) {
        var a, b, i, val = this.value;
        // close any open lists
        closeAllLists();
        if (!val) { return false;}
        currentStatus= -1;
        // create div to contain item values
        a = document.createElement("DIV");
        a.setAttribute("id", this.id + "autocomplete-list");
        a.setAttribute("class", "autocomplete-items");
        // append as child of container
        this.parentNode.appendChild(a);

        for (i = 0; i < arr.length; i++) {
          // check if the item starts with the same letters as what was entered
          if (arr[i].substr(0, val.length).toUpperCase() == val.toUpperCase()) {
            // create div for matching element
            b = document.createElement("DIV");
            // bold matching element
            b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
            b.innerHTML += arr[i].substr(val.length);
            // insert a input field that will hold the current item's value
            b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
            // execute function on click
                b.addEventListener("click", function(e) {
                // insert the value for the input text field
                inp.value = this.getElementsByTagName("input")[0].value;
                // close lists again
                closeAllLists();
            });
            a.appendChild(b);

          }
        }

    });
    //execute on keypress
    inp.addEventListener("keydown", function(e) {
        var x = document.getElementById(this.id + "autocomplete-list");
        if (x) x = x.getElementsByTagName("div");
        if (e.keyCode == 40) {
          // increase current state when down arrow is pressed
          currentState++;
          // make current var more visible
          addActive(x);
        } else if (e.keyCode == 38) { //up
          // On up arrow key, decrease currentState
          currentState--;
          //  make the current item more visible
          addActive(x);
        } else if (e.keyCode == 13) {
          // If enter key is pressed, prevent the form from being submitted
          e.preventDefault();
          if (currentState > -1) {
            // simulate a click on the "active" item
            if (x) x[currentState].click();
          }
        }
    });
    function addActive(x) {
      // function to classify item as active
      if (!x) return false;
      // remove any already existing active classes
      removeActive(x);
      if (currentState >= x.length) currentState = 0;
      if (currentState < 0) currentState = (x.length - 1);
      // add class "autocomplete-active"
      x[currentFocus].classList.add("autocomplete-active");
    }
    function removeActive(x) {
      // Remove the "active" class from all autocomplete items
      for (var i = 0; i < x.length; i++) {
        x[i].classList.remove("autocomplete-active");
      }
    }
    function closeAllLists(elmnt) {
      // close all autocomplete lists in the document, except the one passed as an argument
      var x = document.getElementsByClassName("autocomplete-items");
      for (var i = 0; i < x.length; i++) {
        if (elmnt != x[i] && elmnt != inp) {
        x[i].parentNode.removeChild(x[i]);
      }
    }
   }

   }
