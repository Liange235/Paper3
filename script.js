document.addEventListener("DOMContentLoaded", function() {
  var htmlContentDiv = document.getElementById("html-content");
  var iframe = document.createElement("iframe");
  iframe.setAttribute("src", "https://liange235.github.io/Intermediate-results/1D.html");
  iframe.setAttribute("frameborder", "0");
  iframe.setAttribute("width", "100%");
  iframe.setAttribute("height", "400px");
  htmlContentDiv.appendChild(iframe);
});
