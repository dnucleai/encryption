
// @GENERATOR:play-routes-compiler
// @SOURCE:/Users/MarcWuDunn/Documents/encryption/conf/routes
// @DATE:Tue Jun 19 23:31:03 PDT 2018


package router {
  object RoutesPrefix {
    private var _prefix: String = "/"
    def setPrefix(p: String): Unit = {
      _prefix = p
    }
    def prefix: String = _prefix
    val byNamePrefix: Function0[String] = { () => prefix }
  }
}
