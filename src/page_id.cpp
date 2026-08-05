#include "flashtier/page_id.hpp"

namespace flashtier {

std::string PageId::to_string() const {
    return std::to_string(value);
}

}  // namespace flashtier
